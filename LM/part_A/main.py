import torch.optim as optim

from model import *
from utils import *
from functions import *

# Paths settings
TRAIN_DATA_PATH = "dataset/ptb.train.txt"
DEV_DATA_PATH = "dataset/ptb.valid.txt"
TEST_DATA_PATH = "dataset/ptb.test.txt"
MODELS_PATH = "bin"
LOG_PATH = "experiment_log.csv"
PLOTS_PATH = "plots"

# Default configuration settings
configs = {
    "training": True,
    "use_lstm": False,
    "use_dropout": False,
    "use_adamw": False
}

# Default training hyperparameters
params = {
    "lr": 0.001,
    "hid_size": 200,
    "emb_size": 300,
    "out_dropout": 0.1,
    "emb_dropout": 0.1,
    "tr_batch_size": 64,
    "clip": 5,
    "n_epochs": 100,
    "patience_init": 3
}

if __name__ == "__main__":
    # Prepare data
    train_loader, dev_loader, test_loader, lang, vocab_len = prepare_data(
        TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH, params
    )

    # Select mode and model
    configs = select_config(configs)
    model_filename = f"{get_config(configs)}.pt"
    model_path = os.path.join(MODELS_PATH, model_filename)

    # Define the loss functions
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    if configs["training"]: # Training mode
        # Select the hyperparameters
        params = select_params(params)

        # Iniziatilize the model
        if configs["use_lstm"]:
            model = LM_LSTM(
                params["emb_size"], params["hid_size"], vocab_len,
                params["emb_dropout"], params["out_dropout"],
                configs["use_dropout"], pad_index=lang.word2id["<pad>"]
            ).to(DEVICE)
        else:
            model = LM_RNN(
                params["emb_size"], params["hid_size"], vocab_len,
                pad_index=lang.word2id["<pad>"]
            ).to(DEVICE)

        model.apply(init_weights)

        # Select the optimizer
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"]) if configs["use_adamw"] else optim.SGD(model.parameters(), lr=params["lr"])

        # Train the model
        print("\n==================== Training... ====================")
        results = train_model(
            model, train_loader, dev_loader, test_loader,
            criterion_train, criterion_eval, optimizer, params
        )
        save_model = True
        if os.path.exists(model_path): # Compare with the existing model
            # Load the existing model
            if configs["use_lstm"]:
                ref_model = LM_LSTM(
                    params["emb_size"], params["hid_size"], vocab_len,
                    params["emb_dropout"], params["out_dropout"],
                    configs["use_dropout"], pad_index=lang.word2id["<pad>"]
                ).to(DEVICE)
            else:
                ref_model = LM_RNN(
                    params["emb_size"], params["hid_size"], vocab_len,
                    pad_index=lang.word2id["<pad>"]
                ).to(DEVICE)
            ref_model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            # Evaluate the existing model performances
            ref_model.eval()
            ref_ppl, _ = eval_loop(test_loader, criterion_eval, ref_model)
            
            # Compare the models
            print(f"Comparing models...\nExisting model PPL: {ref_ppl:.2f}")
            if ref_ppl <= results["final_ppl"]:
                save_model = False
                print("Existing model is better or equal. Keeping it.\n")
            else:
                print(f"New model has lower test PPL ({results['final_ppl']:.2f} < {ref_ppl:.2f}). Replacing the model.\n")

        # Save the model if better than the existing one
        if save_model:
            model_data = {
                'model_state_dict': results["best_model"].state_dict(),
                'params': params
            }
            torch.save(model_data, model_path)
            print(f"Saved model and hyperparameters as {model_filename}\n")

        # Log and plot results
        log_results(configs, params, results, LOG_PATH)
        plot_data(configs, results, PLOTS_PATH)
    else: # Testing mode
        if os.path.exists(model_path): # Test the existing model
            # Load the existing model
            print("\nTesting the selected model...\n")
            saved_data = torch.load(model_path, map_location=DEVICE)
            model_state_dict = saved_data['model_state_dict']
            saved_params = saved_data['params']
            params.update(saved_params)

            if configs["use_lstm"]:
                ref_model = LM_LSTM(
                    params["emb_size"], params["hid_size"], vocab_len,
                    params["emb_dropout"], params["out_dropout"],
                    configs["use_dropout"], pad_index=lang.word2id["<pad>"]
                ).to(DEVICE)
            else:
                ref_model = LM_RNN(
                    params["emb_size"], params["hid_size"], vocab_len,
                    pad_index=lang.word2id["<pad>"]
                ).to(DEVICE)
            ref_model.load_state_dict(model_state_dict)

            # Evaluate the existing model performances
            ref_model.eval()
            ref_ppl, _ = eval_loop(test_loader, criterion_eval, ref_model)
            print("\n==================== Test Results ====================")
            print(f"Test PPL of model with {get_config(configs)}: {ref_ppl:.2f}")
            print("=====================================================\n")
        else:
            print(f"\nError: Model {model_filename} not found. Exiting.")
            exit(1)