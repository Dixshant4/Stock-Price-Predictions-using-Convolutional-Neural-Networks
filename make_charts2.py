from copy import deepcopy

import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import random
import os


# Fetching sufficient data to calculate the 100-day moving average
tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)
# Assuming '6mo' is enough to cover the 100-day period, adjust if necessary
tickerDf = tickerData.history(period='25y')

# Calculate the 5-day and 100-day Moving Averages of the Close prices
tickerDf['5d_MA'] = tickerDf['Close'].rolling(window=5).mean()
tickerDf['100d_MA'] = tickerDf['Close'].rolling(window=100).mean()

# Trim the DataFrame to the last 20 days for the plot
last_20_days_df = tickerDf.tail(20)

print("length of data")
print(len(tickerDf))


slices = []
for i in range(0, len(tickerDf) - 20, 1):
    slice_ = tickerDf.iloc[i:i+20]
    slices.append(slice_)



# Got rid of the first 99 slices because all of them had NaN values for their 100 day moving average
slices = slices[99:]

data_with_labels = []
for i in range(len(slices) - 1):
    # labels = {buy = 1, hold = 0, sell = -1}
    # curr_data_with_label = [data, label]
    curr_data_with_label = [slices[i]]  # First insert the data
    close_curr_day = slices[i]['Close'][-1]
    close_next_day = slices[i + 1]['Close'][0]
    threshold = 1.005
    if close_curr_day > close_next_day * threshold:
        curr_data_with_label.append(-1)  # Sell
    elif close_curr_day < close_next_day * threshold:
        curr_data_with_label.append(1)  # Buy
    else:
        curr_data_with_label.append(0)  # Hold
    data_with_labels.append(curr_data_with_label)


def plot_and_save_to_pngs(slices):
    for i in range(len(slices)):
        curr_slice = slices[i]
        # Plot a candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=curr_slice.index,
                                             open=curr_slice['Open'],
                                             high=curr_slice['High'],
                                             low=curr_slice['Low'],
                                             close=curr_slice['Close'],
                                             increasing_line_color='green', decreasing_line_color='red',
                                             name="Candlestick", showlegend=False)])
        # Adding the 5-day Moving Average line
        fig.add_trace(go.Scatter(x=curr_slice.index, y=curr_slice['5d_MA'],
                                 mode='lines', name='5-Day MA',
                                 line=dict(color='blue', width=2),
                                 showlegend=False))  # Remove from legend

        # Adding the 100-day Moving Average line
        fig.add_trace(go.Scatter(x=curr_slice.index, y=curr_slice['100d_MA'],
                                 mode='lines', name='100-Day MA',
                                 line=dict(color='orange', width=2),
                                 showlegend=False))  # Remove from legend

        fig.update_layout(xaxis_rangeslider_visible=False,  # Hide the range slider
                          yaxis_showticklabels=False,  # Hide y-axis labels
                          xaxis_showticklabels=False,  # Hide x-axis labels
                          plot_bgcolor='black',  # Set background color to black
                          paper_bgcolor='black',  # Set paper background color to black
                          xaxis=dict(showgrid=False),  # Hide x-axis grid lines
                          yaxis=dict(showgrid=False)  # Hide y-axis grid lines
                          )  # Set font color to white
                          # font = dict(color='white')
        # fig.show()
        path_to_write_image = 'visual_data/fig' + str(i) + '.png'
        fig.write_image(path_to_write_image)


def png_to_numpy_colored(image_path):
    # Open the image
    img = Image.open(image_path)

    # Resize the image to match the size of PneumoniaMNIST dataset
    img_resized = img.resize((128, 128))

    # Convert the image to a numpy array
    img_array = np.array(img_resized)

    # Normalize the pixel values (0-255) to (0-1)
    img_array = img_array / 255.0

    return img_array


def make_background_black(image_matrix, threshold=0.95):
    # Create a copy of the image matrix to avoid modifying the original
    modified_image = image_matrix.copy()

    # If the image has RGBA channels, set the alpha channel to 1 (fully opaque)
    if modified_image.shape[2] == 4:
        modified_image[:, :, 3][modified_image[:, :, 3] != 0] = 1

    # Identify pixels where RGB values are all close to 1 (white background)
    white_pixels = np.all(modified_image[:, :, :3] > 0.9, axis=-1)

    # Set the RGB values of white pixels to 0 (black background)
    modified_image[:, :, :3][white_pixels] = 0

    return modified_image
    # Alternative
    # # Create a copy of the image matrix to avoid modifying the original
    # modified_image = image_matrix.copy()
    #
    # # If the image has RGBA channels, set the alpha channel to 1 (fully opaque)
    # if modified_image.shape[2] == 4:
    #     modified_image[:, :, 3][modified_image[:, :, 3] != 0] = 1
    #
    # # Identify pixels where RGB values are all close to 1 (white background)
    # white_pixels = np.all(modified_image[:, :, :3] > threshold, axis=-1)
    #
    # # Set the RGB values of white pixels to 0 (black background)
    # modified_image[:, :, :3][white_pixels] = 0
    #
    # return modified_image


def count_files_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Count the number of files
    num_files = len(files)

    return num_files


def convert_pngs_to_numpy_matrices(folder_path, data_labels):
    files = os.listdir(folder_path)
    labeled_image_matrices = []
    index = 0
    for file in files:
        if index >= len(data_labels):
            break
        # curr_matrix_with_label = [image_matrix, label]
        # label = {buy:1, sel: -1, hold:0}
        curr_matrix_with_label = []
        curr_matrix = png_to_numpy_colored('visual_data/' + file)  # png converted to numpy matrix
        # curr_matrix = make_background_black(curr_matrix)
        curr_matrix_with_label.append(curr_matrix)  # add the numpy matrix
        curr_matrix_with_label.append(data_labels[index][1])  # add the label of that matrix
        index += 1

        labeled_image_matrices.append(curr_matrix_with_label)
        # plt.imshow(curr_matrix)
        # plt.show()
    return labeled_image_matrices


def save_to_npz(folder_path, data, name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Save the numpy array into a npz file in the folder
    np.save(folder_path + '/' + name, data)


# C:\Users\kerim\PycharmProjects\CSC413\fig1.png
# data_folder = 'C:/Users/kerim/PycharmProjects/CSC413/visual_data'
# image_name = '/fig' + str(i + 1)
# image_path = data_folder + image_name + '.png'
# image_matrix = png_to_numpy_colored(image_path)
# image_matrix = make_background_black(image_matrix)
#
# print("img matrix shape")
# print(image_matrix.shape)  # Output shape should be (28, 28, 3) for RGB images
# plt.imshow(image_matrix)
# plt.show()
#
# print("matrix itself")
# print(image_matrix)
#
# print("last day")
# print(tickerDf.tail(1))
#
# print("len slices")
# print(len(slices))
#
# print("data")
# print(len(data_with_labels))
#
# print("test")
# print("curr")
# print(slices[1098]['Close'][-1])
# print("next")
# print(slices[1099]['Close'][0])
#
# print("label")
# print(data_with_labels[1098][1])


# Keep this commented!!!
# plot_and_save_to_pngs(slices=slices)

# Keep these commented!!!
# image_matrices_with_labels_ordered = convert_pngs_to_numpy_matrices(folder_path='visual_data',
#                                                                    data_labels=data_with_labels)
# image_matrices_with_labels = deepcopy(image_matrices_with_labels_ordered)

# Shuffles the sample
# random.shuffle(image_matrices_with_labels)

# convert to numpy matrix
# image_matrices_with_labels = np.array(image_matrices_with_labels)

# Data Legend:
# image_matrices_with_labels[0][0] --> numpy matrix representation of figure 0
# image_matrices_with_labels[0][1] --> label of figure 0


# save_to_npz(folder_path='numpy_matrix_data', data=image_matrices_with_labels_ordered, name='ordered_data')
# save_to_npz(folder_path='numpy_matrix_data', data=image_matrices_with_labels, name='shuffled_data')
#
# plt.imshow(image_matrices_with_labels[0][0])
# plt.show()


