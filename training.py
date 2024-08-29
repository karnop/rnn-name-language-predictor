import random
import utils
import encoding
import torch
import time, math
import rnn 
import encoding
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def randomchoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomchoice(utils.all_categories)
    line = randomchoice(utils.category_lines[category])
    category_tensor = torch.tensor([utils.all_categories.index(category)], dtype=torch.long)
    line_tensor = encoding.lineToTensor(line)
    return category, line, category_tensor, line_tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return utils.all_categories[category_i], category_i

# for i in range(10):
#    category, line, category_tensor, line_tensor = randomTrainingExample()
#    print(f"category : {category}, line : {line}") 

# training
n_iters = 100000
print_every = 5000
plot_every = 1000

# keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return "%dm %ds " % (m,s)

start = time.time()

# ____________________________________ TRAINING
for iter in range(1, n_iters+1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = rnn.train(category_tensor, line_tensor)
    current_loss += loss

    # print iteration number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = "✓" if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
    
plt.figure()
plt.plot(all_losses)
plt.show()


# ____________________________________ TESTING
# results
confusion = torch.zeros(len(utils.all_categories),len(utils.all_categories))
n_confusion = 10000


# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = rnn.evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = utils.all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(len(utils.all_categories)):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + utils.all_categories, rotation=90)
ax.set_yticklabels([''] + utils.all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = rnn.evaluate(encoding.lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, utils.all_categories[category_index]))
            predictions.append([value, utils.all_categories[category_index]])

predict('Manav')
predict('Aditya')
predict('Abdus')
predict('Kartik')
predict('Bhatt')
predict('Jackson')
predict('Satoshi')
