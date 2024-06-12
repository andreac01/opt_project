import os
import pickle
import matplotlib.pyplot as plt

def list_pickle_files(directory):
    """List all .pickle files in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith('.pickle')]

def load_and_show_pickle_file(filepath):
    """Load and display a pickle file containing a Matplotlib figure."""
    with open(filepath, 'rb') as f:
        fig = pickle.load(f)
    fig.show()

def main():
	directory = './plots/'

	while True:
		pickle_files = []
		for root, dirs, files in os.walk(directory):
			for file in files:
				if file.endswith('.pickle'):
					pickle_files.append(os.path.join(root, file))
		pickle_files = sorted(pickle_files)
		
		if not pickle_files:
			print("No .pickle files found in the directory.")
			break
		
		print("Available .pickle files:")
		for idx, filename in enumerate(pickle_files):
			print(f"{idx + 1}: {filename}")
		
		try:
			choice = int(input("Enter the number of the image you want to open (or 0 to exit): "))
			
			if choice == 0:
				print("Exiting...")
				break
			
			if 1 <= choice <= len(pickle_files):
				chosen_file = pickle_files[choice - 1]
				load_and_show_pickle_file(chosen_file)
			else:
				print("Invalid choice. Please try again.")
		
		except ValueError:
			print("Invalid input. Please enter a number.")
		
		input("Press Enter to continue...")
		plt.close('all')
		os.system('clear')

if __name__ == "__main__":
    main()
