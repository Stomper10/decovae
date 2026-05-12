#!/usr/bin/env bash
# Per-user running GPU usage (handles multi-node uneven GRES like b[11-12] with 2+4 GPUs)

squeue -t R -h -o "%u %i" | while read -r user job; do
  # 1) 가장 정확: 노드별 GRES 상세 합산
  gpus=$(scontrol show job -dd "$job" | awk '
    BEGIN{tot=0}
    /GresDetail=/{
      s=$0
      # 각 괄호 블록 (b11:gpu:2) (b12:gpu:4) ... 파싱
      while (match(s, /\([^)]*\)/)) {
        blk=substr(s, RSTART+1, RLENGTH-2)
        # gpu(:type)?:COUNT 형태 잡기 (예: gpu:rtx3090:2 또는 gpu:2)
        if (match(blk, /gpu(:[^:]+)?:([0-9]+)/, m)) tot+=m[2]
        s=substr(s, RSTART+RLENGTH)
      }
    }
    END{print tot}
  ')

  # 2) 폴백: AllocTRES의 총 gpu 개수(gres/gpu=TOTAL)
  if [ -z "$gpus" ] || [ "$gpus" -eq 0 ] 2>/dev/null; then
    gpus=$(scontrol show job "$job" | awk -F'[, ]' '
      {for(i=1;i<=NF;i++) if($i ~ /^gres\/gpu=/){split($i,a,"="); print a[2]}}
    ')
  fi

  # 3) 마지막 폴백: GRES의 per-node 수 × 노드 수
  if [ -z "$gpus" ] || [ "$gpus" -eq 0 ] 2>/dev/null; then
    read -r gres nodes <<<"$(squeue -h -j "$job" -o "%b %D")"
    pernode=$(awk -v t="$gres" 'BEGIN{n=split(t,a,":"); print a[n]+0}')
    gpus=$(( pernode * nodes ))
  fi

  echo "$user $gpus"
done | awk '{gpu[$1]+=$2} END{for(u in gpu) printf "%-12s %d\n", u, gpu[u]}'
