import pygame
import sys
import time
import  os
import datetime
from othello_game import OthelloGame
#from ai_opt import get_best_move
from ai_agent import get_best_move

# Constants and colors
WIDTH, HEIGHT = 480, 560
BOARD_SIZE = 8
SQUARE_SIZE = (HEIGHT - 80) // BOARD_SIZE
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 128, 0)


class OthelloGUI:
    def __init__(self, player_mode="friend",black_ai=None, white_ai=None):
        """
        A graphical user interface (GUI) for playing the Othello game.

        Args:
            player_mode (str): The mode of the game, either "friend" or "ai" (default is "friend").
        """
        self.win = self.initialize_pygame()
        self.game = OthelloGame(player_mode=player_mode)
        self.message_font = pygame.font.SysFont(None, 24)
        self.message = ""
        self.invalid_move_message = ""
        self.black_ai = black_ai
        self.white_ai = white_ai
        self.ai_thinking_times = {"black": [], "white": []}  # 记录思考时间

        # 创建截图目录
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def take_screenshot(self, winner):
        """截图并保存"""

        if winner == 1:
            winner_name = "black"
        elif winner == -1:
            winner_name = "white"
        else:
            winner_name = "tie"

        # 获取AI名称
        black_ai_name = self.black_ai.__name__ if self.black_ai else "human"
        white_ai_name = self.white_ai.__name__ if self.white_ai else "human"

        filename = f"{self.screenshot_dir}/othello_{black_ai_name}_vs_{white_ai_name}_{winner_name}.png"

        # 截图
        pygame.image.save(self.win, filename)
        print(f"Screenshot saved: {filename}")
        return filename

    def initialize_pygame(self):
        """
        Initialize the Pygame library and create the game window.

        Returns:
            pygame.Surface: The Pygame surface representing the game window.
        """
        pygame.init()
        win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Othello")
        return win

    def draw_board(self):
        """
        Draw the Othello game board and messaging area on the window.
        """
        self.win.fill(GREEN_COLOR)

        # Draw board grid and disks
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pygame.draw.rect(
                    self.win,
                    BLACK_COLOR,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    1,
                )
                if self.game.board[row][col] == 1:
                    pygame.draw.circle(
                        self.win,
                        BLACK_COLOR,
                        ((col + 0.5) * SQUARE_SIZE, (row + 0.5) * SQUARE_SIZE),
                        SQUARE_SIZE // 2 - 4,
                    )
                elif self.game.board[row][col] == -1:
                    pygame.draw.circle(
                        self.win,
                        WHITE_COLOR,
                        ((col + 0.5) * SQUARE_SIZE, (row + 0.5) * SQUARE_SIZE),
                        SQUARE_SIZE // 2 - 4,
                    )

        # Draw messaging area
        message_area_rect = pygame.Rect(
            0, BOARD_SIZE * SQUARE_SIZE, WIDTH, HEIGHT - (BOARD_SIZE * SQUARE_SIZE)
        )
        pygame.draw.rect(self.win, WHITE_COLOR, message_area_rect)

        # Draw player's turn message
        player_turn = "Black's" if self.game.current_player == 1 else "White's"
        turn_message = f"{player_turn} turn"
        message_surface = self.message_font.render(turn_message, True, BLACK_COLOR)
        message_rect = message_surface.get_rect(
            center=(WIDTH // 2, (HEIGHT + BOARD_SIZE * SQUARE_SIZE) // 2 - 20)
        )
        self.win.blit(message_surface, message_rect)

        # Draw invalid move message
        if self.message:
            invalid_move_message = self.message
            message_surface = self.message_font.render(
                invalid_move_message, True, BLACK_COLOR
            )
            message_rect = message_surface.get_rect(
                center=(WIDTH // 2, (HEIGHT + BOARD_SIZE * SQUARE_SIZE) // 2 + 20)
            )
            self.win.blit(message_surface, message_rect)

        # Draw invalid move message
        if self.invalid_move_message:
            message_surface = self.message_font.render(
                self.invalid_move_message, True, BLACK_COLOR
            )
            message_rect = message_surface.get_rect(
                center=(WIDTH // 2, (HEIGHT + BOARD_SIZE * SQUARE_SIZE) // 2 + 20)
            )
            self.win.blit(message_surface, message_rect)

        pygame.display.update()

    def handle_input(self):
        """
        Handle user input events such as mouse clicks and game quitting.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // SQUARE_SIZE
                row = y // SQUARE_SIZE
                if self.game.is_valid_move(row, col):
                    self.game.make_move(row, col)
                    self.invalid_move_message = (
                        ""  # Clear any previous invalid move message
                    )
                else:
                    self.invalid_move_message = "Invalid move! Try again."

    def run_game(self, return_to_menu_callback=None):
        """
        Run the main game loop until the game is over and display the result.
        """

        round_count = 0

        while not self.game.is_game_over():
            self.handle_input()
            current_player_color = "black" if self.game.current_player == 1 else "white"

            # If it's the AI player's turn
            if self.game.player_mode == "ai" and self.game.current_player == -1:
                round_count+=1
                self.message = "AI is thinking..."
                self.draw_board()  # Display the thinking message

                start_time = time.time()

                ai_move = get_best_move(self.game)

                end_time = time.time()
                think_time = end_time - start_time
                print(f"第{round_count}轮，AI思考时间: {think_time:.2f}秒")

                pygame.time.delay(500)  # Wait for a short time to show the message
                if ai_move is not None:
                    self.game.make_move(*ai_move)

            elif self.game.player_mode == "ai_vs_ai":
                round_count += 1
                ai_function = self.black_ai if self.game.current_player == 1 else self.white_ai

                if ai_function:
                    self.message = f"{current_player_color.capitalize()} AI is thinking..."
                    self.draw_board()

                    start_time = time.time()
                    ai_move = ai_function(self.game)
                    end_time = time.time()

                    think_time = end_time - start_time
                    self.ai_thinking_times[current_player_color].append(think_time)
                    print(f"Round {round_count}, {current_player_color} AI thinking time: {think_time:.2f}s")

                    pygame.time.delay(300)  # 短暂延迟以便观察
                    if ai_move is not None:
                        self.game.make_move(*ai_move)

            self.message = ""  # Clear any previous messages
            self.draw_board()

        winner = self.game.get_winner()
        if winner == 1:
            self.message = "Black wins!"
        elif winner == -1:
            self.message = "White wins!"
        else:
            self.message = "It's a tie!"

        self.draw_board()

        screenshot_path = self.take_screenshot(winner)
        print(f"截图已保存至{screenshot_path}")

        pygame.time.delay(5000)

        # Call the return_to_menu_callback if provided
        if return_to_menu_callback:
            return_to_menu_callback()


def run_game():
    """
    Start and run the Othello game.
    """
    othello_gui = OthelloGUI()
    othello_gui.run_game()
