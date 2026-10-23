#!/bin/bash
set -e

cd "$(dirname "$0")"

flutter run \
  --dart-define=SERVER_URL=http://91.144.140.47 \
  --dart-define=API_KEY="dea09d0924f93f636bed17e05a93895d0a1590769b7aa43bda86d20ac8863353"
