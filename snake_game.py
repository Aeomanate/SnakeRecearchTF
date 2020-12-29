import curses
from random import randint as rand


class SnakeGame:
    def __init__(self, board_width=20, board_height=20, gui=False):
        self.score = 0
        self.steps = 0
        self.is_game_end = False
        self.board = {'width': board_width, 'height': board_height}
        self.gui = gui

    def start(self):
        self.snake_init()
        self.generate_food()
        if self.gui: self.gui_init()
        return self.game_state()

    def snake_init(self):
        x = rand(5, self.board["width"] - 5)
        y = rand(5, self.board["height"] - 5)
        self.snake = []
        vertical = rand(0, 1) == 0
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]
            self.snake.insert(0, point)

    def generate_food(self):
        food = []
        while food == []:
            food = [rand(1, self.board["width"]), rand(1, self.board["height"])]
            if food in self.snake: food = []
        self.food = food

    def gui_init(self):
        curses.initscr()
        win = curses.newwin(self.board["width"] + 2, self.board["height"] + 2, 0, 0)
        curses.curs_set(0)
        win.nodelay(1)
        win.timeout(1)
        self.win = win
        self.render()

    def render(self):
        self.win.clear()
        self.win.border(0)
        self.win.addstr(0, 2, 'Sc: ' + str(self.score) + ' ' + 'St: ' + str(self.steps) + ' ')
        self.win.addch(self.food[0], self.food[1], '*')
        for i, point in enumerate(self.snake):
            if i == 0:
                self.win.addch(point[0], point[1], '0')
            else:
                self.win.addch(point[0], point[1], '1')
        self.win.getch()

    def step(self, key):
        # 0 - UP   1 - RIGHT 2 - DOWN 3 - LEFT
        if self.is_game_end: self.end_game()
        self.create_new_point(key)
        if self.food_eaten():
            self.score += 1
            self.generate_food()
        else:
            self.remove_tail()
        self.check_self_collision()
        self.steps += 1
        if self.gui: self.render()
        return self.game_state()

    def create_new_point(self, key):
        new_point = [self.snake[0][0], self.snake[0][1]]
        if key == 0:
            new_point[0] -= 1
        elif key == 1:
            new_point[1] += 1
        elif key == 2:
            new_point[0] += 1
        elif key == 3:
            new_point[1] -= 1
        self.snake.insert(0, new_point)

    def remove_tail(self):
        self.snake.pop()

    def food_eaten(self):
        return self.snake[0] == self.food

    def check_self_collision(self):
        if (self.snake[0][0] == 0 or
                self.snake[0][0] == self.board["width"] + 1 or
                self.snake[0][1] == 0 or
                self.snake[0][1] == self.board["height"] + 1 or
                self.snake[0] in self.snake[1:]):
            self.is_game_end = True

    def game_state(self):
        return self.is_game_end, self.score, self.snake, self.food, self.steps

    @staticmethod
    def gui_destroy():
        curses.endwin()

    def end_game(self):
        if self.gui: self.gui_destroy()
        raise Exception("Game over")


if __name__ == "__main__":
    game = SnakeGame(gui=True)
    game.start()
    for _ in range(200):
        if game.step(rand(0, 3))[0]:
            break
