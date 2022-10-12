# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation dataset creation functions."""

import csv
import functools
import os
from os import path
import pickle
import random
from typing import Iterable, Mapping, Union

from absl import logging
import chex
import jax
import jax.numpy as jnp
from kubric.challenges.point_tracking import dataset
import mediapy as media
import numpy as np
from PIL import Image
from scipy import io
import tensorflow as tf
import tensorflow_datasets as tfds

from tapnet import tapnet_model
from tapnet.utils import transforms

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]


# TODO(doersch): can we remove the jax dependency?
def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, chex.Array]:
  """Package a set of frames and tracks for use in TAPNet evaluations.

  Given a set of frames and tracks with no query points, sample queries
  strided every query_stride frames, ignoring points that are not visible
  at the selected frames.

  Args:
    target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
      where True indicates occluded.
    target_points: Position, of shape [n_tracks, n_frames, 2], where each point
      is [x,y] scaled between 0 and 1.
    frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
      -1 and 1.
    query_stride: When sampling query points, search for un-occluded points
      every query_stride frames and convert each one into a query.

  Returns:
    A dict with the keys:
      video: Video tensor of shape [1, n_frames, height, width, 3].
      query_points: Query points of shape [1, n_queries, 3] where
        each point is [t, y, x] scaled to the range [-1, 1].
      target_points: Target points of shape [1, n_queries, n_frames, 2] where
        each point is [x, y] scaled to the range [-1, 1].
      trackgroup: Index of the original track that each query point was
        sampled from.  This is useful for visualization.
  """
  tracks = []
  occs = []
  queries = []
  trackgroups = []
  total = 0
  trackgroup = np.arange(target_occluded.shape[0])
  for i in range(0, target_occluded.shape[1], query_stride):
    mask = target_occluded[:, i] == 0
    query = jnp.stack(
        [
            i * jnp.ones(target_occluded.shape[0:1]), target_points[:, i, 1],
            target_points[:, i, 0]
        ],
        axis=-1,
    )
    queries.append(query[mask])
    tracks.append(target_points[mask])
    occs.append(target_occluded[mask])
    trackgroups.append(trackgroup[mask])
    total += np.array(jnp.sum(target_occluded[:, i] == 0))

  return {
      'video':
          frames[jnp.newaxis, ...],
      'query_points':
          jnp.concatenate(queries, axis=0)[jnp.newaxis, ...],
      'target_points':
          jnp.concatenate(tracks, axis=0)[jnp.newaxis, ...],
      'occluded':
          jnp.concatenate(occs, axis=0)[jnp.newaxis, ...],
      'trackgroup':
          jnp.concatenate(trackgroups, axis=0)[jnp.newaxis, ...],
  }


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, chex.Array]:
  """Package a set of frames and tracks for use in TAPNet evaluations.

  Given a set of frames and tracks with no query points, use the first
  visible point in each track as the query.

  Args:
    target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
      where True indicates occluded.
    target_points: Position, of shape [n_tracks, n_frames, 2], where each point
      is [x,y] scaled between 0 and 1.
    frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
      -1 and 1.

  Returns:
    A dict with the keys:
      video: Video tensor of shape [1, n_frames, height, width, 3]
      query_points: Query points of shape [1, n_queries, 3] where
        each point is [t, y, x] scaled to the range [-1, 1]
      target_points: Target points of shape [1, n_queries, n_frames, 2] where
        each point is [x, y] scaled to the range [-1, 1]
  """

  valid = np.sum(~target_occluded, axis=1) > 0
  target_points = target_points[valid, :]
  target_occluded = target_occluded[valid, :]

  query_points = []
  for i in range(target_points.shape[0]):
    index = np.where(target_occluded[i] == 0)[0][0]
    x, y = target_points[i, index, 0], target_points[i, index, 1]
    query_points.append(np.array([index, y, x]))  # [t, y, x]
  query_points = np.stack(query_points, axis=0)

  return {
      'video': frames[np.newaxis, ...],
      'query_points': query_points[np.newaxis, ...],
      'target_points': target_points[np.newaxis, ...],
      'occluded': target_occluded[np.newaxis, ...],
  }


def create_jhmdb_dataset(jhmdb_path: str) -> Iterable[DatasetElement]:
  """JHMDB dataset, including fields required for PCK evaluation."""
  gt_dir = jhmdb_path
  videos = []
  for file in tf.io.gfile.listdir(path.join(gt_dir, 'splits')):
    # JHMDB file containing the first split, which is standard for this type of
    # evaluation.
    if not file.endswith('split1.txt'):
      continue

    video_folder = '_'.join(file.split('_')[:-2])
    for video in tf.io.gfile.GFile(path.join(gt_dir, 'splits', file), 'r'):
      video, traintest = video.split()
      video, _ = video.split('.')

      traintest = int(traintest)
      video_path = path.join(video_folder, video)

      if traintest == 2:
        videos.append(video_path)

  # Shuffle so numbers converge faster.
  random.shuffle(videos)

  for video in videos:
    logging.info(video)
    joints = path.join(gt_dir, 'joint_positions', video, 'joint_positions.mat')

    if not tf.io.gfile.exists(joints):
      logging.info('skip %s', video)
      continue

    gt_pose = io.loadmat(tf.io.gfile.GFile(joints, 'rb'))['pos_img']
    gt_pose = np.transpose(gt_pose, [1, 2, 0])
    frames = path.join(gt_dir, 'Rename_Images', video, '*.png')
    framefil = tf.io.gfile.glob(frames)
    framefil.sort()

    def read_frame(f):
      im = Image.open(tf.io.gfile.GFile(f, 'rb'))
      im = im.convert('RGB')
      return np.array(im.getdata()).reshape([im.size[1], im.size[0], 3])

    frames = [read_frame(x) for x in framefil]
    frames = np.stack(frames)
    num_frames, height, width, _ = frames.shape
    invalid_x = np.logical_or(
        gt_pose[:, 0:1, 0] < 0,
        gt_pose[:, 0:1, 0] >= width,
    )
    invalid_y = np.logical_or(
        gt_pose[:, 0:1, 1] < 0,
        gt_pose[:, 0:1, 1] >= height,
    )
    invalid = np.logical_or(invalid_x, invalid_y)
    invalid = np.tile(invalid, [1, gt_pose.shape[1]])
    invalid = invalid[:, :, jnp.newaxis].astype(np.float32)
    gt_pose_orig = gt_pose

    gt_pose = transforms.convert_grid_coordinates(
        gt_pose,
        np.array([width, height]),
        np.array(tapnet_model.TRAIN_SIZE[2:0:-1]),
    )
    # Set invalid poses to -1 (outside the frame)
    gt_pose = (1. - invalid) * gt_pose + invalid * (-1.)

    frames = np.array(
        jax.jit(
            functools.partial(
                jax.image.resize,
                shape=[num_frames, *tapnet_model.TRAIN_SIZE[1:4]],
                method='bilinear',
            ))(frames))
    frames = frames / (255. / 2.) - 1.
    queries = gt_pose[:, 0]
    queries = np.concatenate(
        [queries[..., 0:1] * 0, queries[..., ::-1]],
        axis=-1,
    )
    if gt_pose.shape[1] < frames.shape[0]:
      # Some videos have pose sequences that are shorter than the frame
      # sequence (usually because the person disappears).  In this case,
      # truncate the video.
      logging.warning('short video!!')
      frames = frames[:gt_pose.shape[1]]

    converted = {
        'video': frames[np.newaxis, ...],
        'query_points': queries[np.newaxis, ...],
        'target_points': gt_pose[np.newaxis, ...],
        'gt_pose': gt_pose[np.newaxis, ...],
        'gt_pose_orig': gt_pose_orig[np.newaxis, ...],
        'occluded': gt_pose[np.newaxis, ..., 0] * 0,
        'fname': video,
        'im_size': np.array([height, width]),
    }
    yield {'jhmdb': converted}


def create_kubric_eval_train_dataset(
    mode: str,
    max_dataset_size: int = 100,
) -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on Kubric training data."""
  res = dataset.create_point_tracking_dataset(
      split='train',
      train_size=tapnet_model.TRAIN_SIZE[1:3],
      batch_dims=[1],
      shuffle_buffer_size=None,
      repeat=False,
      vflip='vflip' in mode,
      random_crop=False)

  num_returned = 0

  for data in res[0]():
    if num_returned >= max_dataset_size:
      break
    num_returned += 1
    yield {'kubric': data}


def create_kubric_eval_dataset(mode: str) -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on Kubric val data."""
  res = dataset.create_point_tracking_dataset(
      split='validation',
      batch_dims=[1],
      shuffle_buffer_size=None,
      repeat=False,
      vflip='vflip' in mode,
      random_crop=False,
  )
  np_ds = tfds.as_numpy(res)

  for data in np_ds:
    yield {'kubric': data}


def create_davis_dataset(
    davis_points_path: str,
    query_mode: str = 'strided') -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on DAVIS data."""
  pickle_path = davis_points_path

  with tf.io.gfile.GFile(pickle_path, 'rb') as f:
    davis_points_dataset = pickle.load(f)

  for video_name in davis_points_dataset:
    frames = davis_points_dataset[video_name]['video']
    num_frames = frames.shape[0]
    # Use jit to avoid leaking gpu memory
    frames = np.array(
        jax.jit(
            functools.partial(
                jax.image.resize,
                shape=[
                    num_frames,
                    tapnet_model.TRAIN_SIZE[1],
                    tapnet_model.TRAIN_SIZE[2],
                    3,
                ],
                method='bilinear'))(frames))
    frames = frames.astype(np.float32) / 255. * 2. - 1.
    target_points = davis_points_dataset[video_name]['points']
    target_occ = davis_points_dataset[video_name]['occluded']
    target_points *= np.array([
        tapnet_model.TRAIN_SIZE[2],
        tapnet_model.TRAIN_SIZE[1],
    ])

    if query_mode == 'strided':
      converted = sample_queries_strided(target_occ, target_points, frames)
    elif query_mode == 'first':
      converted = sample_queries_first(target_occ, target_points, frames)
    else:
      raise ValueError(f'Unknown query mode {query_mode}.')

    yield {'davis': converted}


def create_rgb_stacking_dataset(
    robotics_points_path: str,
    query_mode: str = 'strided') -> Iterable[DatasetElement]:
  """Dataset for evaluating performance on robotics data."""
  pickle_path = robotics_points_path

  with tf.io.gfile.GFile(pickle_path, 'rb') as f:
    robotics_points_dataset = pickle.load(f)

  for example in robotics_points_dataset:
    frames = example['video']
    frames = frames.astype(np.float32) / 255. * 2. - 1.
    target_points = example['points']
    target_occ = example['occluded']
    target_points *= np.array(
        [tapnet_model.TRAIN_SIZE[2], tapnet_model.TRAIN_SIZE[1]])

    if query_mode == 'strided':
      converted = sample_queries_strided(target_occ, target_points, frames)
    elif query_mode == 'first':
      converted = sample_queries_first(target_occ, target_points, frames)
    else:
      raise ValueError(f'Unknown query mode {query_mode}.')

    yield {'robotics': converted}


def create_kinetics_dataset(
    kinetics_path: str,
    query_mode: str = 'strided') -> Iterable[DatasetElement]:
  """Kinetics point tracking dataset."""
  csv_path = path.join(kinetics_path, 'tapvid_kinetics.csv')

  point_tracks_all = dict()
  with tf.io.gfile.GFile(csv_path) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      youtube_id = row[0]
      point_tracks = np.array(row[3:]).reshape(-1, 3)
      if youtube_id in point_tracks_all:
        point_tracks_all[youtube_id].append(point_tracks)
      else:
        point_tracks_all[youtube_id] = [point_tracks]

  for video_id in point_tracks_all:
    video_path = path.join(kinetics_path, 'videos', video_id + '_valid.mp4')
    frames = media.read_video(video_path)
    frames = media.resize_video(frames, tapnet_model.TRAIN_SIZE[1:3])
    frames = frames.astype(np.float32) / 255. * 2. - 1.

    point_tracks = np.stack(point_tracks_all[video_id], axis=0)
    point_tracks = point_tracks.astype(np.float32)
    if frames.shape[0] < point_tracks.shape[1]:
      logging.info('Warning: short video!')
      point_tracks = point_tracks[:, :frames.shape[0]]
    point_tracks, occluded = point_tracks[..., 0:2], point_tracks[..., 2]
    occluded = occluded > 0
    target_points = point_tracks * np.array(
        [tapnet_model.TRAIN_SIZE[2], tapnet_model.TRAIN_SIZE[1]])

    if query_mode == 'strided':
      converted = sample_queries_strided(occluded, target_points, frames)
    elif query_mode == 'first':
      converted = sample_queries_first(occluded, target_points, frames)
    else:
      raise ValueError(f'Unknown query mode {query_mode}.')

    yield {'kinetics': converted}