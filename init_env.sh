#!/usr/bin/env bash
export TPAE_WORKSPACE_PATH="$(pwd)/TPAE_workspace"
echo $TPAE_WORKSPACE_PATH
cd $TPAE_WORKSPACE_PATH;tpae_init_workspace;cd ..
