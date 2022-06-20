#!/bin/bash

COMMIT_SIZE=$(python get_size.py)

if [[ $COMMIT_SIZE -le 100 ]]
then
    echo $COMMIT_SIZE
else
    echo "File/Folder size exceeds GitHub's file limit of 100 MB"
fi

# git status