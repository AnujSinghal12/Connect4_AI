from collections import defaultdict
from math import sqrt
import math
import numpy as np
import random
import copy
import time
import gzip
import json
from matplotlib import pyplot as plt


class Board:
    def __init__(self, rows, columns):
        self.grid = np.zeros((rows, columns), dtype=int)
        self.rows = rows
        self.columns = columns

    def move(self, column, player):
        row = self.rows - 1
        if self.is_valid_move(column):
            for i in range(self.rows):
                if self.grid[i][column] != 0:
                    row = i-1
                    break
            self.grid[row, column] = player
        if self.is_win() != 0:
            return self.is_win()
        else:
            return 0

    def is_valid_move(self, column):
        return self.grid[0][column] == 0

    def get_all_valid_moves(self):
        return [i for i in range(self.grid.shape[1]) if self.is_valid_move(i)]

    def is_win(self):

        draw_value = 3
        if get_all_valid_moves(self) == []:
            return draw_value

        for i in range(self.rows):
            for j in range(self.columns):
                if(i+3 < self.rows):
                    if (self.grid[i][j] != 0) and (self.grid[i][j] == self.grid[i+1][j]) and (self.grid[i+1][j] == self.grid[i+2][j]) and (self.grid[i+2][j] == self.grid[i+3][j]):
                        return self.grid[i][j]
                if(j+3 < self.columns):
                    if (self.grid[i][j] != 0) and (self.grid[i][j] == self.grid[i][j+1]) and (self.grid[i][j+1] == self.grid[i][j+2]) and (self.grid[i][j+2] == self.grid[i][j+3]):
                        return self.grid[i][j]

        for i in range(self.rows - 3):
            for j in range(self.columns - 3):
                if (self.grid[i][j] != 0) and (self.grid[i][j] == self.grid[i+1][j+1]) and (self.grid[i+1][j+1] == self.grid[i+2][j+2]) and (self.grid[i+2][j+2] == self.grid[i+3][j+3]):
                    return self.grid[i][j]

        for i in range(3, self.rows):
            for j in range(self.columns - 3):
                if (self.grid[i][j] != 0) and (self.grid[i][j] == self.grid[i-1][j+1]) and (self.grid[i-1][j+1] == self.grid[i-2][j+2]) and (self.grid[i-2][j+2] == self.grid[i-3][j+3]):
                    return self.grid[i][j]

        return 0

    def get_state(self):
        return self.grid


def get_all_valid_moves(board):
    type(board.grid)
    return [i for i in range(board.grid.shape[1]) if is_valid_move(board.grid, i)]


def is_valid_move(grid, column):
    return grid[0][column] == 0


class Node:
    def __init__(self, board, parent, terminal, move, c, victory, player):
        self.wins = 0
        self.games = 0
        self.board = board
        self.parent = parent
        self.terminal = terminal
        self.move = move
        self.c = c
        self.children = None
        self.victory = victory
        self.player = player

    def set_children(self, children):
        self.children = children

    def get_value(self):
        if self.games == 0:
            return math.inf
        return self.wins/self.games + 1*sqrt(math.log(self.parent.games)/self.games)

    def game_get_value(self):
        if self.games == 0:
            return 0
        return self.wins/self.games

    def set_all_valid_moves(self):
        self.all_valid_moves = self.board.get_all_valid_moves()

    def is_terminal(self):
        return self.terminal

    def get_board(self):
        return self.board


def switch_player(player):
    if player == 1:
        player = 2
    else:
        player = 1
    return player


def mcts(tree, no_of_simulations, player):
    original_player = player
    c = 1
    draw_value = 3
    for i in range(no_of_simulations):
        # selection
        if tree is None:
            tree = Node(Board(6, 5), None, 0, None, c, 0, player)
        who_won = 0
        node = tree
        if node.is_terminal() == 0:
            while node.children is not None:
                next_node = None
                next_node_value = 0
                epsilon = 0.8
                for child in node.children:
                    if child.is_terminal() != 0:
                        next_node = child
                        who_won = child.is_terminal()
                        break
                    if child.get_value() >= next_node_value:
                        next_node = child
                        next_node_value = next_node.get_value()
                    if random.uniform(0, 1) > epsilon:
                        next_node = random.choice(node.children)
                node = next_node

        # expansion

        if node.is_terminal() == 0:
            all_valid_moves = get_all_valid_moves(node.get_board())
            children = []

            for move in all_valid_moves:
                old_board = copy.deepcopy(node.board)
                terminal = old_board.move(move, node.player)
                parent = node
                children.append(Node(old_board, parent, terminal,
                                move, c, 0, switch_player(node.player)))
            node.set_children(children)

            for child in node.children:
                if child.is_terminal() != 0:
                    node = child
                    who_won = child.is_terminal()
                    break
            if who_won == 0 and node.children != None:
                node = random.choice(node.children)

            player = switch_player(player)

        # simulation

        terminal = node.is_terminal()
        simulation_board = copy.deepcopy(node.get_board())
        backprop_node = node

        while terminal == 0:
            all_valid_moves = get_all_valid_moves(simulation_board)
            if all_valid_moves == []:
                break
            move = random.choice(all_valid_moves)
            terminal = simulation_board.move(move, player)
            if terminal != 0:
                who_won = terminal

            player = switch_player(player)

        # backprop

        backprop_node.games += 1
        if who_won == original_player:
            backprop_node.wins += 1
        if who_won == draw_value:
            backprop_node.wins += 0.2
        backprop_node = backprop_node.parent
        while backprop_node is not None:
            backprop_node.games += 1
            if who_won == original_player:
                backprop_node.wins += 1
            if who_won == draw_value:
                backprop_node.wins += 0.2
            backprop_node = backprop_node.parent

    i = 0
    max_child_value = 0
    best_child = None
    for child in tree.children:
        if child.get_value() >= max_child_value:
            max_child_value = child.get_value()
            best_child = child
    return best_child, max_child_value


def get_max_q_state(state, q):
    if str(state.grid.tobytes()) not in q.keys():
        return 0
    else:
        max_q_value = -math.inf
        for q_value in q[str(state.grid.tobytes())]:
            if max_q_value < q_value:
                max_q_value = q_value
        return max_q_value


def PrintGrid(positions):
    print('\n'.join(' '.join(str(x) for x in row) for row in positions))
    print()


if __name__ == '__main__':

    inp = input("Enter \'0\' for MCTS or \'1\' for Q learning")
    if inp == '1':
        playmcts = False
        playq = True
    else:
        playmcts = True
        playq = False

    if playmcts:
        win1 = 0
        win2 = 0
        draw = 0
        # for i in range(50):
        game_board = Board(6, 5)

        turn_of = 1
        total_moves = 0
        while game_board.is_win() == 0:
            if turn_of == 1:
                move1, game_max_child_value = mcts(
                    Node(game_board, None, 0, None, 1, 0, 1), 40, 1)
                game_board.move(move1.move, 1)
                turn_of = 2
                print("Action selected: ", move1.move)
                print('Value of next state according to MCTS : ',
                      game_max_child_value)
                total_moves += 1
            elif turn_of == 2:
                move2, game_max_child_value = mcts(
                    Node(game_board, None, 0, None, 1, 0, 2), 200, 2)
                game_board.move(move2.move, 2)
                turn_of = 1
                print("Action selected: ", move2.move)
                print('Value of next state according to MCTS : ',
                      game_max_child_value)
                total_moves += 1
            PrintGrid(game_board.grid)

        if game_board.is_win() == 3:
            print('Game Drawn, Total moves =', total_moves)
        else:
            print('Player', game_board.is_win(),
                  'has WON. Total moves =', total_moves)
        print()

    if playq:
        q = dict()
        # with gzip.open('./dataFile.dat.gz', 'wb') as f:
        #     f.write(bytes(json.dumps(q), 'utf-8'))
        alpha = 0.4
        gamma = 0.8

        board = Board(4, 5)
        while board.is_win() == 0:
            # with gzip.open('./dataFile.dat.gz', 'rb') as f:
            #     file = f.read()
            #     q = json.loads(file.decode('utf-8'))

            move1, game_max_child_value = mcts(
                Node(board, None, 0, None, 1, 0, 1), 40, 1)
            board.move(move1.move, 1)
            print('Action selected : ', move1.move)
            print('Value of next state according to MCTS : ', game_max_child_value)
            print(board.grid)

            old_state = copy.deepcopy(board)
            if str(board.grid.tobytes()) not in q.keys():
                q[str(old_state.grid.tobytes())] = dict()

            all_valid_moves = board.get_all_valid_moves()
            next_state = None
            greedy_move = 0
            for move in all_valid_moves:
                if move not in q[str(board.grid.tobytes())].keys():
                    q[str(board.grid.tobytes())][move] = 0
                    if greedy_move <= q[str(board.grid.tobytes())][move]:
                        greedy_move = q[str(board.grid.tobytes())][move]
            terminal = 0
            for move in all_valid_moves:
                if greedy_move == q[str(board.grid.tobytes())][move]:
                    terminal = board.move(move, 2)
                    selected_move = move
                    break
            if terminal == 1:
                reward = 100
            elif terminal == 2:
                reward = -100
            elif terminal == 3:
                reward = 10
            else:
                reward = 0

            print('Action selected : ', selected_move)
            print('Value of next state according to QLearning : ',
                  q[str(old_state.grid.tobytes())][selected_move])
            print(board.grid)

            q[str(old_state.grid.tobytes())][selected_move] += alpha * \
                (reward + gamma*(get_max_q_state(board, q)) -
                    q[str(old_state.grid.tobytes())][selected_move])

            with gzip.open('./dataFile.dat.gz', 'wb') as f:
                f.write(bytes(json.dumps(q), 'utf-8'))

            if q.get(str(board.grid.tobytes()), None) == None:
                q[board] = dict()

        if(board.is_win() == 3):
            print("Game Draw")
        else:
            print("Player", board.is_win(), "has won.")
