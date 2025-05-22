import torch.optim as optim

from utils import prepare_data
from functions import *

# Paths settings
TRAIN_DATA_PATH = "dataset/ptb.train.txt"
DEV_DATA_PATH = "dataset/ptb.valid.txt"
TEST_DATA_PATH = "dataset/ptb.test.txt"
MODELS_PATH = "bin"
LOG_PATH = "testing_results/experiments_log.csv"
PLOT_PATH = "testing_results/plots"

# Default configuration settings
configs = {
    "training": True,
    "use_lstm": False,
    "use_dropout": False,
    "use_adamw": False
}

# Default training hyperparameters
params = {
    "lr": 0.0001,
    "hid_size": 200,
    "emb_size": 300,
    "out_dropout": 0.3,
    "emb_dropout": 0.3,
    "tr_batch_size": 64,
    "clip": 5,
    "n_epochs": 100,
    "patience_init": 3
}

if __name__ == "__main__":
    # Select mode and model
    select_config(configs)
    model_filename = f"{get_config(configs)}.pt"
    model_path = os.path.join(MODELS_PATH, model_filename)

    # Prepare data
    train_loader, dev_loader, test_loader, lang, vocab_len = prepare_data(
        TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH, params
    )

    # Define the loss functions
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    if configs["training"]: # Training mode
        # Select the hyperparameters
        select_params(params) 

        # Iniziatilize the model
        model = init_model(lang, vocab_len, params, configs)
        model.apply(init_weights)
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"]) if configs["use_adamw"] else optim.SGD(model.parameters(), lr=params["lr"])

        # Train the model
        results = train_model(
            model, train_loader, dev_loader, test_loader,
            criterion_train, criterion_eval, optimizer, params
        )

        # Save the model
        os.makedirs(MODELS_PATH, exist_ok=True)
        model_data = {
            'model_state_dict': results["best_model"].state_dict(),
            'params': params
        }
        torch.save(model_data, model_path)
        print(f"Saved model and hyperparameters as {model_filename}\n")

        # Log and plot results
        log_and_plot_results(configs, params, results, LOG_PATH, PLOT_PATH)

    else: # Testing mode
        if os.path.exists(model_path):
            # Load the existing model
            ref_model = load_model(model_path, lang, vocab_len, configs)

            # Evaluate the existing model performances
            ref_ppl, _ = eval_loop(test_loader, criterion_eval, ref_model)

            # Show results
            print("\n==================== Test Results ====================")
            print(f"Test PPL of model with {get_config(configs)}: {ref_ppl:.2f}")
            print("=====================================================\n")
        else:
            print(f"\nError: Model {model_filename} not found. Exiting.")
            exit(1)