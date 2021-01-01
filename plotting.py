import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
# sphinx_gallery_thumbnail_number = 2

# f = open('p0_strat.pkl', 'rb')
# raw_data = pickle.load(f)
# f.close()
def np2img(raw_data):
    strat = np.zeros((13,13,3))
    for datum in raw_data.keys():
        hrank, lrank, suited = datum
        action = 2
        if suited:
            strat[12-hrank][12-lrank] = (raw_data[datum][0], raw_data[datum][2], raw_data[datum][1])
        else:
            strat[12-lrank][12-hrank] = (raw_data[datum][0], raw_data[datum][2], raw_data[datum][1])

    fig, ax = plt.subplots()
    im = ax.imshow(strat)

    # # We want to show all ticks...
    ax.set_xticks(np.arange(13))
    ax.set_yticks(np.arange(13))
    ax.xaxis.tick_top()
    ax.set_xticklabels(list('AKQJT') + list(range(9,1,-1)))
    ax.set_yticklabels(list('AKQJT') + list(range(9,1,-1)))
    ax.set_xlabel('suited')
    ax.set_ylabel('unsuited')
    ax.xaxis.set_label_position('top')


    ax.set_title("Action Probability 012->RBG")
    fig.tight_layout()
    plt.savefig('p0_strat_img.png')