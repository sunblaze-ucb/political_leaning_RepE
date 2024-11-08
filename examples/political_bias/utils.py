import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

def political_dataset(user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> (list, list):
    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv('political_statement.csv')
    statement=df['statement'].tolist()
    template_str1 = "Pretend to be a politically independent person. Please describe whether the following sentence should be true or not within two sentences: {s}"
    template_str2 = "Pretend to be a typical Democrat. Please describe whether the following sentence should be true or not within two sentences: {s}"
    rep=[]
    dem=[]
    for st in statement:
        rep.append(f"{user_tag} {template_str1.format(s=st)} {assistant_tag} As a politically independent person, I")
        dem.append(f"{user_tag} {template_str2.format(s=st)} {assistant_tag} As a typical Democrat, I")
    combined_data = [[r,d] for r,d in zip(rep,dem)]
    train_data = combined_data

    train_labels = []
    for td in train_data:
        rep_s = td[0]
        random.shuffle(td)
        train_labels.append([s == rep_s for s in td])
    
    train_data = np.concatenate(train_data).tolist() 
    
    # Create test data
    reshaped_data = np.array([[r,d] for r,d in zip(rep, dem)]).flatten()
    test_data = reshaped_data.tolist() 

    # Load the data
    df = pd.read_csv('topics.csv')
    statement=df['topic'].tolist()
    template_str1 = "You're simulating an Independent candidate. Please make a speech regarding {s} within two sentences."
    template_str2 = "You're simulating a Democrat candidate. Please make a speech regarding {s} within two sentences."
    rep=[]
    dem=[]
    for st in statement:
        rep.append(f"{user_tag} {template_str1.format(s=st)} {assistant_tag} As an Independent candidate, I")
        dem.append(f"{user_tag} {template_str2.format(s=st)} {assistant_tag} As a Democrat candidate, I")
    combined_data = [[r,d] for r,d in zip(rep,dem)]
    train_data2 = combined_data

    train_labels2 = []
    for td in train_data2:
        rep_s = td[0]
        random.shuffle(td)
        train_labels2.append([s == rep_s for s in td])
    
    train_labels+=train_labels2
    train_data2 = np.concatenate(train_data2).tolist() 
    train_data+=train_data2

    # Create test data
    reshaped_data2 = np.array([[r,d] for r,d in zip(rep, dem)]).flatten()
    test_data2 = reshaped_data2.tolist()
    test_data+=test_data2

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }


def plot_lat_scans(input_ids, rep_reader_scores_dict, layer_slice):
    for rep, scores in rep_reader_scores_dict.items():
        start_tok = input_ids.index('Ġassistant') #index('/') #. #index('▁A') #
        print(start_tok, np.array(scores).shape)
        standardized_scores = np.array(scores)[start_tok+50:start_tok+100,layer_slice]
        # print(standardized_scores.shape)

        bound = np.mean(standardized_scores) + np.std(standardized_scores)
        bound = 2.3

        # standardized_scores = np.array(scores)
        
        threshold = 0
        standardized_scores[np.abs(standardized_scores) < threshold] = 1
        standardized_scores = standardized_scores.clip(-bound, bound)
        
        cmap = 'coolwarm'

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        sns.heatmap(-standardized_scores.T, cmap=cmap, linewidth=0.5, annot=False, fmt=".3f", vmin=-bound, vmax=bound)
        ax.tick_params(axis='y', rotation=0)

        ax.set_xlabel("Token Position")#, fontsize=20)
        ax.set_ylabel("Layer")#, fontsize=20)

        # x label appear every 5 ticks

        ax.set_xticks(np.arange(0, len(standardized_scores), 5)[1:])
        ax.set_xticklabels(np.arange(0, len(standardized_scores), 5)[1:])#, fontsize=20)
        ax.tick_params(axis='x', rotation=0)

        ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
        ax.set_yticklabels(np.arange(20, len(standardized_scores[0])+20, 5)[::-1][1:])#, fontsize=20)
        ax.set_title("LAT Neural Activity")#, fontsize=30)
    plt.savefig('neural_activity.png')

def plot_detection_results(input_ids, rep_reader_scores_dict, THRESHOLD, start_answer_token=":"):

    cmap=LinearSegmentedColormap.from_list('rg',["r", (255/255, 255/255, 224/255), "g"], N=256)
    colormap = cmap

    # Define words and their colors
    words = [token.replace('▁', ' ') for token in input_ids]

    # Create a new figure
    fig, ax = plt.subplots(figsize=(12.8, 10), dpi=200)

    # Set limits for the x and y axes
    xlim = 1000
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 10)

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Starting position of the words in the plot
    x_start, y_start = 1, 8
    y_pad = 0.3
    # Initialize positions and maximum line width
    x, y = x_start, y_start
    max_line_width = xlim

    y_pad = 0.3
    word_width = 0

    iter = 0

    selected_concepts = ["pol"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):

        rep_scores = np.array(rep_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean+5*std) | (rep_scores < mean-5*std)] = mean # get rid of outliers
        mag = max(0.3, np.abs(rep_scores).std() / 10)
        min_val, max_val = -mag, mag
        norm = Normalize(vmin=min_val, vmax=max_val)

        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD # change this for threshold
            rep_scores = rep_scores / np.std(rep_scores[5:])
            rep_scores = np.clip(rep_scores, -mag, mag)
        if "flip" in n_style:
            rep_scores = -rep_scores
        
        rep_scores[np.abs(rep_scores) < 0.0] = 0

        # ofs = 0
        # rep_scores = np.array([rep_scores[max(0, i-ofs):min(len(rep_scores), i+ofs)].mean() for i in range(len(rep_scores))]) # add smoothing
        
        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)


        # Initialize positions and maximum line width
        x, y = x_start, y_start
        max_line_width = xlim
        started = False
            
        for word, score in zip(words[5:], rep_scores[5:]):

            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue
            
            color = colormap(norm(score))

            # Check if the current word would exceed the maximum line width
            if x + word_width > max_line_width:
                # Move to next line
                x = x_start
                y -= 3

            # Compute the width of the current word
            text = ax.text(x, y, word, fontsize=13)
            word_width = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width
            word_height = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).height

            # Remove the previous text
            if iter:
                text.remove()

            # Add the text with background color
            text = ax.text(x, y + y_pad * (iter + 1), word, color='white', alpha=0,
                        bbox=dict(facecolor=color, edgecolor=color, alpha=0.8, boxstyle=f'round,pad=0', linewidth=0),
                        fontsize=13)
            
            # Update the x position for the next word
            x += word_width + 0.1
        
        iter += 1