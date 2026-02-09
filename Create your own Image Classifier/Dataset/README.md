# 📦 Data Preparation Script

This folder contains the automation logic for the project's dataset.

### Why is this here?
Instead of requiring you to manually download and sort 300MB of flower images, the `dataset.py` script handles the environment setup for you.

### What `dataset.py` does:
1. **Downloads**: Fetches the `flower_data.tar.gz` from AWS.
2. **Extracts**: Unpacks the images into a structured `/flowers` directory.
3. **Maps**: Creates `cat_to_name.json` which links the folder numbers to real flower names.
4. **Cleans**: Deletes the downloaded `.tar.gz` file to save your disk space.

### How to run it:
From your terminal, run:
`python dataset.py`
