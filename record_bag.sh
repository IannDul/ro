#!/usr/bin/env bash

set -e

RUN_NAME="test_run_$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$HOME/ros2_test_runs/$RUN_NAME"

mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/logs"

export ROS_LOG_DIR="$RUN_DIR/logs"

echo "Run dir: $RUN_DIR"

# Служебная информация до запуска
{
  echo "Date: $(date)"
  echo "User: $(whoami)"
  echo "Host: $(hostname)"
  echo "ROS_DISTRO: $ROS_DISTRO"
  echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
} > "$RUN_DIR/info.txt"

# Запускаем твой launch и одновременно пишем весь вывод терминала в console.log
ros2 launch my_package my_launch.py 2>&1 | tee "$RUN_DIR/console.log" &

LAUNCH_PID=$!

# Даём нодам немного времени подняться
sleep 3

# Сохраняем список нод и топиков
ros2 node list > "$RUN_DIR/node_list.txt" || true
ros2 topic list -t > "$RUN_DIR/topic_list.txt" || true

# Пишем все топики
ros2 bag record -a \
  --include-hidden-topics \
  --compression-mode file \
  --compression-format zstd \
  -o "$RUN_DIR/bag"

# Когда остановишь rosbag через Ctrl+C, остановим launch
kill "$LAUNCH_PID" || true

echo "Saved to: $RUN_DIR"