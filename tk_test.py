import tkinter as tk
import sys

root = tk.Tk()
root.title("Tkinter test")
root.geometry("320x120+100+100")
try:
    root.attributes('-topmost', True)
except Exception:
    pass

lbl = tk.Label(root, text="Tkinter test window — will close in 3s", font=("Segoe UI", 10))
lbl.pack(padx=12, pady=28)

def close():
    print("TK: closing after timeout")
    root.destroy()

root.after(3000, close)
print("TK: starting mainloop")
root.mainloop()
print("TK: mainloop exited")
