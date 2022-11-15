from tkinter import Frame, Label, CENTER, Button
import threading
import numpy as np
import math
import copy
import torch

import config as c
import game_logic as logic
from agent.agent import Agent


class GameGrid(Frame):
    def __init__(self):
        frame = Frame.__init__(self)
        self.step_count = 0
        self.grid()
        self.master.title('2048')

        # A button to start auto-solve
        Button(frame, text='DQN-train', command=self.start_dqn_thread).grid(column=1, row=0)
        Button(frame, text='Refresh', command=self.refresh).grid(column=1, row=2)

        self.master.bind("<Key>", self.key_down)

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)

        # DQN items
        num_states = c.GRID_LEN * c.GRID_LEN
        num_actions = len(c.ACTION_NUMBERS)
        self.agent = Agent(num_states, num_actions)

        # start main loop
        self.update_grid_cells()
        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def refresh(self):
        self.step_count = 0
        self.matrix = logic.new_game(c.GRID_LEN)
        self.update_grid_cells()

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT:
            exit()
        elif key in self.commands:
            self.commit_move(key)

    def commit_move(self, key):

        self.matrix, done, step_score = self.commands[key](self.matrix)
        # Print the score

        if done:
            self.step_count += 1
            self.matrix = logic.add_two_or_four(self.matrix)
            '''
            print("The monotone score: " + str(logic.score_monotone(self.matrix)) +
                  " square amount score: " + str(logic.score_number_of_squares(self.matrix)) +
                  " weighted square amount score: " + str(logic.score_weighted_squares(self.matrix)))
            '''
            # record last move
            self.update_grid_cells()
            state = logic.game_state(self.matrix)

            if state != 'not over':
                if state == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                elif state == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

        return step_score, done

    def start_dqn_thread(self):
        thread = threading.Thread(target=self.auto_solve, args=())
        thread.setDaemon(True)
        thread.start()

    def auto_solve(self):
        print('DQN solution...')
        complete_episodes = 0

        # Each episode:
        for episode in range(c.NUM_EPISODES):
            # Reset the game:
            observation, status, action_l = self.reset()

            # Prepare the state in tensor[[]]
            state = torch.from_numpy(observation).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            # Take steps till the end of the game.
            step = 0
            while status == 'not over':
                action = self.agent.get_action(state, action_l, episode)

                # Commit the step: tensor.item converts the tensor with a single item to a single value.
                state_next, status, action_l, step_score, effective = self.step(action.item())

                reward = torch.LongTensor([[math.log2(int(max(step_score, 1)))]])

                if effective:
                    step += 1

                if status == 'not over':
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                elif status == 'win':
                    state_next = None
                    complete_episodes += 1

                elif status == 'lose':
                    state_next = None
                    complete_episodes = 0

                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_q_function()
                state = state_next

                if status == 'win' or status == 'lose':
                    print('%d Episode: Finished after %d steps in status %s.' % (episode, step, status))
                    if episode % c.TARGET_NET_UPDATE_INTERVAL == 0:
                        self.agent.update_target_q_function()
                    break

            if complete_episodes >= c.ACCEPT_THRESHOLD:
                print('{} successful episodes.'.format(c.ACCEPT_THRESHOLD))
                self.agent.save_network()
                break

    # environment functions
    # The game should have functions:
    # reset: re-initiate the game
    # observe: get the current game situation, including the number of each cell, whether the game is over
    #   and the possible actions
    # step: make a step in the game. Should return the status.
    def reset(self):
        self.refresh()
        return self.observe()

    # actions: 0 up, 1 down, 2 left, 3 right Should return the status.
    def step(self, action):
        key = c.ACTION_NUMBERS.get(action, None)

        score, is_effective = self.commit_move(key)

        observation, status, action_l = self.observe()

        return observation, status, action_l, score, is_effective

    def observe(self):
        # number of each cell
        observation = np.array(self.matrix).flatten()
        observation = np.array(list(map(lambda x: math.log2(max(x, 1)), observation)))
        # game is over or not. 'win', 'lose', 'not over'
        status = logic.game_state(self.matrix)
        # possible actions
        matrix_copy = copy.deepcopy(self.matrix)

        action_l = logic.get_possible_actions(matrix_copy)
        return observation, status, action_l


if __name__ == "__main__":
    game_grid = GameGrid()
