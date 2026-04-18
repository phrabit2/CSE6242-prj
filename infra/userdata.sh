#!/bin/bash
set -e

# 시스템 업데이트
apt-get update -y
apt-get upgrade -y

# Python 환경 설치
apt-get install -y python3-pip python3-venv git

# 프로젝트 디렉토리 생성
mkdir -p /home/ubuntu/project
cd /home/ubuntu/project

# Python 가상환경 생성 및 라이브러리 설치
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install ruptures          # Change-Point Detection
pip install plotly dash       # 대시보드
pip install pandas numpy      # 데이터 처리
pip install scipy statsmodels # 통계 분석
pip install boto3             # AWS S3 연동
pip install pybaseball        # Statcast 데이터 수집

# GitHub 레포 클론
git clone https://github.com/phrabit2/CSE6242-prj.git /home/ubuntu/project/repo

# 소유권 설정
chown -R ubuntu:ubuntu /home/ubuntu/project