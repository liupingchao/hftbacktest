#!/bin/bash
set -euxo pipefail

exec > >(tee /var/log/ec2-latency-hunt-bootstrap.log) 2>&1
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl
curl -fsSL \
  -o /tmp/amazon-ssm-agent.deb \
  https://s3.ap-northeast-1.amazonaws.com/amazon-ssm-ap-northeast-1/latest/debian_amd64/amazon-ssm-agent.deb
dpkg -i /tmp/amazon-ssm-agent.deb || apt-get -f install -y
systemctl enable amazon-ssm-agent
systemctl restart amazon-ssm-agent
