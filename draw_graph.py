import matplotlib.pyplot as plt
import numpy as np


def drawG(score, name1, name2):
    plt.style.use('seaborn')
    plt.figure(figsize = (8, 4))
    plt.subplot(polar=True)

    subjects = ['Topology(Connection)', 'Exterior', 'Interior',
                'Topology(Plan)', 'Concept']

    # subjects = ['Topology', 'Exterior Image(SSIM)', 'Exterior Image(MSE)', 'Interior Image(SSIM)', 'Interior Image(MSE)', 'Concept(Text)',\
    #             'Plan Image(SSIM)', 'Plan Image(MSE)']
    (lines, labels) = plt.thetagrids(range(0, 360, int(360/len(subjects))), (subjects))

    theta = np.linspace(0, 2 * np.pi, len(score))
    plt.plot(theta, score)
    plt.fill(theta, score, 'b', alpha=0.2)

    plt.savefig(f'./temp_result/{name1}&{name2}.png')


