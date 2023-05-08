#!/bin/bash

if [ $# -ge 1 ]; then
    name=$1
else
name="upload"
fi


# clear the cache
git rm -r --cached

# add all the files to commit
git add .

# commit changes
git commit -m $name

# push the changes
git push -u origin main
