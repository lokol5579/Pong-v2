#!/bin/sh

set -x

# compare MixTrain_DuelingDoubleDQN with game robot
python multiplayer.py -memo_1=MixTrain_DuelingDoubleDQN -agent_1=DuelingDoubleDQN -start_episode_1=1999 -player=1 -test_mode

# compare MixTrain_DuelingDoubleDQN with DuelingDoubleDQN
python multiplayer.py -memo_1=MixTrain_DuelingDoubleDQN -agent_1=DuelingDoubleDQN -start_episode_1=1999 -memo_2=DuelingDoubleDQN -agent_2=DuelingDoubleDQN -start_episode_2=364 -player=2 -test_mode

# compare MixTrain_DuelingDoubleDQN with DuelingDQN
python multiplayer.py -memo_1=MixTrain_DuelingDoubleDQN -agent_1=DuelingDoubleDQN -start_episode_1=1999 -memo_2=DuelingDQN -agent_2=DuelingDQN -start_episode_2=348 -player=2 -test_mode

# compare MixTrain_DuelingDoubleDQN with DQN
python multiplayer.py -memo_1=MixTrain_DuelingDoubleDQN -agent_1=DuelingDoubleDQN -start_episode_1=1999 -memo_2=DQN -agent_2=DQN -start_episode_2=335 -player=2 -test_mode

# compare MixTrain_DuelingDoubleDQN with DoubleDQN
python multiplayer.py -memo_1=MixTrain_DuelingDoubleDQN -agent_1=DuelingDoubleDQN -start_episode_1=1999 -memo_2=DoubleDQN -agent_2=DoubleDQN -start_episode_2=330 -player=2 -test_mode

set +x