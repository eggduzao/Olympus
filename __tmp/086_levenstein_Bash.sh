#!/bin/bash

levenshtein() {
  local s1="$1"
  local s2="$2"
  local len1=${#s1}
  local len2=${#s2}

  # Initialize a 2D matrix as a 1D array
  local -a dist
  for ((i=0; i<=len1; i++)); do
    for ((j=0; j<=len2; j++)); do
      idx=$((i*(len2+1)+j))
      if ((i==0)); then
        dist[idx]=$j
      elif ((j==0)); then
        dist[idx]=$i
      else
        dist[idx]=0
      fi
    done
  done

  # Dynamic programming fill
  for ((i=1; i<=len1; i++)); do
    for ((j=1; j<=len2; j++)); do
      idx=$((i*(len2+1)+j))
      idx_del=$(((i-1)*(len2+1)+j))
      idx_ins=$((i*(len2+1)+(j-1)))
      idx_sub=$(((i-1)*(len2+1)+(j-1)))

      char1="${s1:i-1:1}"
      char2="${s2:j-1:1}"
      if [[ "$char1" == "$char2" ]]; then
        cost=0
      else
        cost=1
      fi

      del=$((dist[idx_del]+1))
      ins=$((dist[idx_ins]+1))
      sub=$((dist[idx_sub]+cost))

      # take minimum
      min=$del
      [[ $ins -lt $min ]] && min=$ins
      [[ $sub -lt $min ]] && min=$sub

      dist[idx]=$min
    done
  done

  echo "${dist[len1*(len2+1)+len2]}"
}

# Sample Input
s1="kitten"
s2="sitting"
levenshtein "$s1" "$s2"

