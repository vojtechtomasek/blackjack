from main import Blackjack
import numpy as np

class QLearning:
    def __init__(self, gamma = 0.9, alpha = 0.1, epsilon = 0.1):
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate
        self.epsilon = epsilon # exploration parameter
        
        self.q_table = np.zeros((32, 12, 2, 2)) # 32 player sums, 12 dealer upcards, 2 usable aces, 2 actions

    def choose_action(self, player_sum, dealer_card, usable_ace):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(["hit, stay"])
        else:
            if self.q_table[player_sum, dealer_card, usable_ace, 0] > self.q_table[player_sum, dealer_card, usable_ace, 1]:
                return "hit"
            else:
                return "stay"
            
    def update(self, player_sum, dealer_card, usable_ace, action, reward, new_player_sum, new_dealer_card, new_usable_ace):
        action_idx = 0 if action == "hit" else 1
        old_value = self.q_table[player_sum, dealer_card, usable_ace, action_idx]
        future_max = np.max(self.q_table[new_player_sum, new_dealer_card, new_usable_ace])
        self.q_table[player_sum, dealer_card, usable_ace, action_idx] = old_value + self.alpha * (reward + self.gamma * future_max - old_value)
         

    @staticmethod
    def has_usable_ace(hand):
        value = 0
        ace = False

        for card in hand:
            card_number = card['number']
            value += min(10, int(card_number) if card_number not in ['J', 'Q', 'K', 'A'] else 11)
            ace |= (card_number == 'A')
        return int(ace and value + 10 <= 21)
    

    def train(self, episodes):
        one_percent = round(episodes / 100)

        for epis in range(episodes):
            game = Blackjack()
            game.start_game()

            if epis % one_percent == 0:
                progress = (epis / episodes) * 100
                print(f"Training progress: {progress:.2f}%")

            
            dealer_card = int(game.dealer_hand[0]['number']) if game.dealer_hand[0]['number'] not in ['J', 'Q', 'K', 'A'] else (10 if game.dealer_hand[0]['number'] != 'A' else 11)
            status = "continue"

            while status == "continue":
                player_sum = game.hand_value(game.player_hand)
                usable_ace = self.has_usable_ace(game.player_hand)
                action = self.choose_action(player_sum, dealer_card, usable_ace)
                status = game.player_action(action)
                new_player_sum = game.hand_value(game.player_hand)
                new_usable_ace = self.has_usable_ace(game.player_hand)

                reward = 0

                if status == "player_blackjack":
                    reward = 1
                elif status == "player_bust":
                    reward = -1

                if reward != 0:
                    self.update(player_sum, dealer_card, usable_ace, action, reward, new_player_sum, dealer_card, new_usable_ace)

                if action == "stay":
                    break

            
            final_result = game.game_result()
            final_reward = 1 if final_result == "player_win" else (-1 if final_result == "player_loss" else 0)
            self.update(player_sum, dealer_card, usable_ace, action, final_reward, new_player_sum, dealer_card, new_usable_ace)



    def play(self):
        game = Blackjack()
        game.start_game()

        print("Dealer shows:", game.format_cards(game.dealer_hand[:1]))

        status = "continue"
        print(game.format_cards(game.player_hand), game.hand_value(game.player_hand))
        while status == "continue":
            player_sum = game.hand_value(game.player_hand)
            usable_ace = self.has_usable_ace(game.player_hand)
            dealer_card = int(game.dealer_hand[0]['number']) if game.dealer_hand[0]['number'] not in ['J', 'Q', 'K', 'A'] else (10 if game.dealer_hand[0]['number'] != 'A' else 11)
            action = "hit" if self.q_table[player_sum, dealer_card, usable_ace, 0] > self.q_table[player_sum, dealer_card, usable_ace, 1] else "stay"
            status = game.player_action(action)

            if action == "stay":
                break

            print(game.format_cards(game.player_hand), game.hand_value(game.player_hand))

        if status == "continue":
            print("Dealer has: ", game.format_cards(game.dealer_hand), game.hand_value(game.dealer_hand))
            game.dealer_action()

        final_result = game.game_result()

        return final_result
    

agent = QLearning()
agent.train(1_000_000)

test_games = 100_000
wins = 0
losses = 0
draws = 0

for _ in range(test_games):
    print("--------------------")
    result = agent.play()
    print(result)
    if result == "player_win":
        wins += 1
    elif result == "player_loss":
        losses += 1
    else:
        draws += 1

print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
print(f"Win rate: {(wins + 0.5 * draws)/(wins + losses + draws) * 100:.2f}%")


#   50 000 samples
#   10 000 training
#   40,39 %

#   500 000 samples
#   100 000 training
#   39,87 %

#   1 000 000 samples
#   850 000 training
#   40,34 %