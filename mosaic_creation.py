import gradio as gr  #importing gradio
import numpy as np   #importing numpy
from PIL import Image
import time

def process_image(image,grid_size,tile_set):
    start = time.perf_counter()
    if image is None:
        raise gr.Error("Please Upload an image")
    
    processed = resize_image(image,grid_size)
    average_colors,categories = create_grid(processed, grid_size,tile_set) # calling create grid function
    segmented = create_segmented_image(average_colors,processed) # Calling create segmented image function
    mosaic = create_mosaic(processed, categories, grid_size,tile_set) # calling create masaic function
    end = time.perf_counter()
    print(f"Processing time for {grid_size}x{grid_size}: {end - start:.4f} seconds")
    compare_execution_time(image, grid_size, tile_set)
    mse = calculate_mse(processed, mosaic) # calling calculate mse function
    ssim = calculate_ssim(processed, mosaic) # calling ssim function 
    
    return segmented, mosaic, mse, ssim

def resize_image(image,grid_size):   # Resizing image so that the image dimensions are divisible by grid size
    image = Image.fromarray(image)
    image= np.array(image)

    height, width = image.shape[:2]

    new_height = height - height %grid_size
    new_width = width - width %grid_size

    return image[:new_height, :new_width]
    

def create_grid(image,grid_size,tile_set):   #Creating grid of the given image by dividing it into small cells 
    height,width=image.shape[:2]             #where cell size depends on grid_size

    cell_height=height//grid_size
    cell_width= width//grid_size

    grid=image.reshape(grid_size, cell_height, grid_size, cell_width,3)
    average_colors=grid.mean(axis=(1,3))    #Calculating average color of each cell

    color_pallette=np.array([             #Defining your own color pallette for mosaic creation
        [255,0,0],  # 0: Red             #Defining colors using RGB 
        [0,255,0],  # 1: Green
        [0,0,255],  # 2: Blue
        [255,255,0],# 3: Yellow
        [255,165,0],# 4: Orange
        [128,0,128],# 5: Purple
        [255,255,255],# 6: White
        [128,128,128],# 7: Gray
        [0,0,0],      # 8: Black
        [139, 69, 19],   #9: Brown
        [245, 245, 220],  #10: Beige
        [255, 182, 193],  #11: Pink
        [135, 206, 235],  # 12: Sky blue
    ])
    selected_colors=color_pallette[:tile_set]  #Selecting colors on basis of preference of number of tiles[3,6,9,12]
    comparisons=(                   # Comparing color of cells with colors of color palette
        average_colors[:,:,None,:]-
        selected_colors[None,None,:,:]
    )
    distance = np.sum(comparisons**2, axis=3) #Calculating distance using Squared Euclidean distance between each cell average color and palette color
    categories =np.argmin(distance,axis=2) #Finding index of closest palette color
    return average_colors,categories # returning avg color of each cell and the index of closest palette color



def create_segmented_image(average_colors, image): #Shows intermediate stage of image: where each cell represents its avg RGB color
    height,width=image.shape[:2]                # Image dimensions
    grid_size = average_colors.shape[0]         # Number of rows in averag ecolors
    cell_height=height//grid_size               #Calculating cell dimensions
    cell_width= width//grid_size

    segmented = np.repeat(                     # Using np.repeat() to complete the grid, we use if twice first for rows and then columns
        np.repeat(average_colors,cell_height,axis=0),
        cell_width,
        axis=1
    )
    return segmented.astype(np.uint8) #Converting float numbers to integers(of images)


def create_mosaic(image, categories,grid_size,tile_set): #Mapping ecah cell with the closest color palette tile and replacing the cell with tile to create mosaic
    height,width=image.shape[:2]
    cell_height=height//grid_size
    cell_width=width//grid_size
    
    tile_files=["tiles/red.png",    #tiles for defined color palette
                "tiles/green.png",
                "tiles/blue.png",
                "tiles/yellow.jpg",
                "tiles/orange.jpg",
                "tiles/purple.jpg",
                "tiles/white.jpg",
                "tiles/gray.jpg",
                "tiles/black.jpg",
                "tiles/brown.jpg",
                "tiles/beige.jpg",
                "tiles/pink.jpg",
                "tiles/skyblue.jpg",
            ]
    selected_files=tile_files[:tile_set] #number of tiles will be chosen as per user prefernce[3,6,9,12]
    tiles=np.array([np.array(Image.open(fname).convert("RGB").resize((cell_width,cell_height)))
            for fname in selected_files])       #Load and resize the tile images as per the cell size
    selcted_tiles=tiles[categories]      #Select the correct tile for every cell 
    mosaic=selcted_tiles.transpose(0,2,1,3,4).reshape(height,width,3) #Creating final mosaic
    return mosaic

def calculate_mse(processed,mosaic):     #function to calculate Mean Squared Error
    processed = processed.astype(np.float64)
    mosaic = mosaic.astype(np.float64)
    return np.mean((processed-mosaic)**2)

from skimage.metrics import structural_similarity #function to calculate Structural Similarity Index
def calculate_ssim(processed, mosaic):
    return structural_similarity(
        processed,
        mosaic,
        channel_axis=2,
        data_range=255
    )
def create_grid_loop(image, grid_size, tile_set): # Implementing create grid using loops
    height, width = image.shape[:2]

    cell_height = height // grid_size
    cell_width = width // grid_size

    color_palette = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 165, 0],
        [128, 0, 128],
        [255, 255, 255],
        [128, 128, 128],
        [0, 0, 0],
        [139, 69, 19],
        [245, 245, 220],
        [255, 182, 193],
        [135, 206, 235]
    ])

    selected_colors = color_palette[:tile_set]

    average_colors = np.zeros((grid_size, grid_size, 3))
    categories = np.zeros((grid_size, grid_size), dtype=int)

    for row in range(grid_size):
        for col in range(grid_size):

            cell = image[
                row * cell_height:(row + 1) * cell_height,
                col * cell_width:(col + 1) * cell_width
            ]

            average_color = cell.mean(axis=(0, 1))
            average_colors[row, col] = average_color

            min_distance = float("inf")
            closest_color = 0

            for k in range(tile_set):
                distance = np.sum(
                    (average_color - selected_colors[k]) ** 2
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_color = k

            categories[row, col] = closest_color

    return average_colors, categories

def compare_execution_time(image, grid_size, tile_set): # Implementing compare execution time to compare vectorization vs loops
    processed = resize_image(image, grid_size)

    start = time.perf_counter()                      # Vectorized implementation

    avg_vector, cat_vector = create_grid(
        processed, grid_size, tile_set
    )

    vectorized_time = time.perf_counter() - start

    start = time.perf_counter()                       # Loop-based implementation

    avg_loop, cat_loop = create_grid_loop(
        processed, grid_size, tile_set
    )

    loop_time = time.perf_counter() - start
    assert np.allclose(avg_vector, avg_loop)            # Check that both produce same results
    assert np.array_equal(cat_vector, cat_loop)

    print(f"\nGrid size: {grid_size}x{grid_size}")
    print(f"Vectorized time: {vectorized_time:.6f} seconds")
    print(f"Loop-based time: {loop_time:.6f} seconds")
    print(f"Speedup: {loop_time / vectorized_time:.2f}x")


demo = gr.Interface(      #Gradio User Interface to upload image as numpy image under label "Upload Image"
                            #and connecting it to python process_image function
    fn=process_image,
    inputs=[gr.Image(type="numpy",image_mode="RGB",label="Upload Image"),
           gr.Slider(minimum=16,maximum=64,step=16,value=32,label="Grid Size"),
           gr.Slider(minimum=3,maximum=12,step=3,value=9,label="Tile Set"),],
    outputs=[gr.Image(label="Segmented Image"),
    gr.Image(label="Mosaic Image"),
    gr.Number(label="MSE",precision=2),
    gr.Number(label="SSIM", precision=4)
    ],
    submit_btn="Generate Mosaic",
    flagging_mode="never",
    
)
demo.launch()