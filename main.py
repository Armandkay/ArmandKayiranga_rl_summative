from training import dqn_training, ppo_training, a2c_training, reinforce_training

def main():
    print("Choose RL algorithm to train:")
    print("1: DQN")
    print("2: PPO")
    print("3: A2C")
    print("4: REINFORCE")
    choice = input("Enter choice (1-4): ")

    if choice == "1":
        dqn_training.train_dqn()
    elif choice == "2":
        ppo_training.train_ppo()
    elif choice == "3":
        a2c_training.train_a2c()
    elif choice == "4":
        reinforce_training.train_reinforce()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
