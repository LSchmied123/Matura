import tkinter as tk
import matplotlib
import os

def display_board(array):
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    def on_enter(event=None):
        global user_input
        user_input = entry.get()  # assign text
        root.destroy()            # close window

    colors = {
        -1: "white",
         0: "light gray",
         1: "red",
         2: "blue"
    }

    root = tk.Tk()
    root.title("Nine Menn's morris")

    rows = len(array)
    cols = len(array[0])

    entry = tk.Entry(root, font=("Arial", 16))
    canvas = tk.Canvas(root, width=cols*50, height=rows*50, bg="white")
    canvas.pack()
    entry.bind("<Return>", on_enter)

    for i in range(rows):
        for j in range(cols):
            x1, y1 = j*50, i*50
            x2, y2 = x1+50, y1+50
            value = array[i][j]
            color = colors.get(value, "gray")
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="white")

    canvas.pack()
    entry.pack()
    entry.focus_set()

    root.mainloop()

    return user_input

def graph_wins(player1, player2, player1_wins, player2_wins, draws):
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    labels = [player1, player2, 'Draws']
    sizes = [player1_wins, player2_wins, draws]
    colors = ['lightcoral', 'lightskyblue', 'lightgreen']
    explode = (0.1, 0.1, 0.1)

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Game Outcomes')
    plt.axis('equal')
    plt.show()
