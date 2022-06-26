from sqlalchemy import true
import torch 
import random
import numpy as np
from snake import SnakeAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.0005

class agentAction:
    def __init__(self):
        self.numOfGames = 0
        self.epsilon = 0 #control randomness 
        self.gamma = 0 # discount rate 
        self.memory = deque(maxlen=MAX_MEM )
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def getState(self,game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remeber(self,state,action,reward, nextState, gameOver):
        self.memory.append((state, action, reward, nextState, gameOver))

    def trainLongMem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def trainShortMem(self,state,action,reward, nextState, gameOver):
        self.trainer.train_step(state, action, reward, nextState, gameOver)


    def getAction(self, state):
         # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.numOfGames
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plotScore = []
    plotMeanScore = []
    totalScore = 0
    record = 0

    agent = agentAction()
    game = SnakeAI()

    while True:
        prevState = agent.getState(game)
        finalMove = agent.getAction(prevState)
        
        reward,gameOver,score = game.play_step(finalMove)
        newState = agent.getState(game)

        agent.trainShortMem(prevState,finalMove,reward,newState, gameOver)

        agent.remeber(prevState,finalMove,reward,newState, gameOver)

        if gameOver:
            game.reset()
            agent.numOfGames += 1   
            agent.trainLongMem()

            if score > record:
                record = score
                
            print('Game: ', agent.numOfGames, 'Score: ', score, 'Record :', record ) 

            plotScore.append(score)
            totalScore += score
            mean_score = totalScore / agent.numOfGames
            plotMeanScore.append(mean_score)
            plot(plotScore, plotMeanScore)
    
    

if __name__ == '__main__':
    train()