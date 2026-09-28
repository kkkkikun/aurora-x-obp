#!/usr/bin/env bash
# 把正式赛提交项目同步到对外 GitHub 仓库并推送（平台 /compete 用的公开仓库 URL）。
#
#   ./sync_submit_repo.sh            # 同步 + commit + push
#   ./sync_submit_repo.sh --dry-run  # 只看会改什么
#
# 源：agentic-observer-project/（唯一权威副本）
# 目标：submit-repo/（独立 git，已 gitignore 出主仓）→ github.com/kkkkikun/aurora-x-obp
# 排除：.env、__pycache__、*.pyc —— 凭据与缓存永不入库（规则禁止）。
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/agentic-observer-project"
DST="$(cd "$(dirname "$0")" && pwd)/submit-repo"
REPO_URL="https://github.com/kkkkikun/aurora-x-obp"
DRY="${1:-}"

[ -d "$SRC" ] || { echo "缺少源目录 $SRC" >&2; exit 1; }
[ -d "$DST/.git" ] || { echo "缺少目标仓库 $DST（首次用 gh repo create 建）" >&2; exit 1; }
[ "$SRC" != "$DST" ] || { echo "源与目标相同，拒绝执行" >&2; exit 1; }

# --exclude '.git' 必须在：rsync --delete 会把源里没有的 .git / .gitignore 删掉。
rsync -a --delete \
  --exclude '.git' --exclude '.gitignore' \
  --exclude '.env' --exclude '__pycache__/' --exclude '*.pyc' \
  "$SRC/" "$DST/"

# 兜底：同步后 .git 必须还在（防止未来改动重新踩掉 --delete 的坑）
[ -d "$DST/.git" ] || { echo "危险：$DST/.git 在同步中丢失，已中止" >&2; exit 1; }

cd "$DST"
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
  echo "无改动：$REPO_URL 已是最新"
  exit 0
fi

git add -A
if [ "$DRY" = "--dry-run" ]; then
  echo "== 将提交以下改动 =="
  git --no-pager diff --cached --stat
  exit 0
fi

git -c user.name="kkkkikun" -c user.email="mikupromax@outlook.com" commit -q -m "同步提交项目：$(date +%F) 更新"
git push -q origin main
echo "已推送 $(git rev-parse --short HEAD) → $REPO_URL"
