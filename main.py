import tkinter as tk
from tkinter import ttk 
from tkinter import messagebox
import sys # Importing sys, though not strictly required, is good practice for main entry files

#+++++++++++++++ Designing the windows+++++++++++++++++++
backgroundColour = '#282c34'
foregroundColour = '#c6c6c6'
accentColour = '#61afef'
buttonBg = '#3c424a'
buttonHover = '#4a5260'

# Attempt to import all module windows needed by topicAction
try:
    from APM1513 import APM1513Window
    from MAT1503 import MAT1503Window

except ImportError as e:
    # This block handles the case where not all module files have been created yet,
    # preventing a crash at startup.
    print(f"Warning: Could not import module classes. Missing file? {e}")

class moduleWindow(tk.Toplevel):
    def __init__(self, parent, moduleCode, moduleName, moduleTopic):
        # This initializes the window
        super().__init__(parent)
        
        # Store module code for reference
        self.moduleCode = moduleCode

        self.title(f"{moduleCode} - {moduleName}")
        self.geometry("1100x700")
        self.configure(bg=backgroundColour)

        # Disable the main menu while this window is open
        parent.attributes('-disabled', True)

        # Protocol to run when the user closes the window
        self.protocol("WM_DELETE_WINDOW", lambda: self.on_close(parent)) 

        # Header section
        headerFrame = tk.Frame(self, bg=backgroundColour)
        headerFrame.pack(pady=20, padx=20, fill="x")
        
        tk.Label(
            headerFrame,
            text=f"{moduleCode}: {moduleName}",
            font=("Arial", 18, "bold"),
            bg=backgroundColour,
            fg=accentColour
        ).pack()
        
        tk.Label(
            headerFrame,
            text="Select a topic to begin studying",
            font=("Arial", 10),
            bg=backgroundColour,
            fg=foregroundColour
        ).pack(pady=(5, 0))

        # Creating a window frame for the topic buttons
        windowFrame = tk.Frame(self, bg=backgroundColour)
        windowFrame.pack(padx=30, pady=10, fill="both", expand=True)

        # Creating the buttons for the chapters/topics in the module 
        for topic in moduleTopic:
            tempButton = tk.Button(
                windowFrame,
                text=topic,
                command=lambda t=topic: self.topicAction(t),
                bg=buttonBg,
                fg=foregroundColour,
                font=("Arial", 11),
                relief='flat',
                cursor='hand2',
                pady=12,
                anchor='w',
                padx=20
            )
            
            # Add hover effects
            tempButton.bind("<Enter>", lambda e: e.widget.config(bg=accentColour, fg='white'))
            tempButton.bind("<Leave>", lambda e: e.widget.config(bg=buttonBg, fg=foregroundColour))
            
            tempButton.pack(fill="x", pady=6)

        # Back button at the bottom
        backButton = tk.Button(
            self,
            text="← Back to Main Menu",
            command=lambda: self.on_close(parent),
            bg=backgroundColour,
            fg=accentColour,
            font=("Arial", 10),
            relief='flat',
            cursor='hand2',
            pady=8
        )
        backButton.pack(pady=15)
        backButton.bind("<Enter>", lambda e: e.widget.config(fg='white'))
        backButton.bind("<Leave>", lambda e: e.widget.config(fg=accentColour))

    def topicAction(self, chapterName):
        # This function launches the correct module window class
        
        if self.moduleCode == "APM1513":
            # APM1513 is Linear Algebra/Octave Code Generation
            APM1513Window(self, chapterName)
            
            
        elif self.moduleCode == "MAT1503":
            # MAT1503 is Linear Algebra/Cramer's Rule/Vectors
            MAT1503Window(self, chapterName)


    def on_close(self, parent):
        # Re-enable the main window
        parent.attributes('-disabled', False)
        self.destroy()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                MAIN CLASS
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class pocketStudentSuite(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Study Assistant Suite - Main Menu")
        self.geometry("1100x700")
        self.configure(bg=backgroundColour)

        # COME BACK HERE TO ADD MODULE CHAPTERS
        self.courseInformation = {
            "APM1513": {"name": "Applied Linear Algebra", "chapters": [
                "Matrix Properties and Manipulation",
                "Solving Square Linear Systems",
                "Eigenvalues and Eigenvectors",
                "Overdetermined Systems and Least Squares",
                "Underdetermined Systems and Null Space",
                "Linear Programming"
            ]},

            "MAT1503": {"name": "Linear Algebra", "chapters": [
                "Systems of Linear Equations & Matrices",
                "Determinants",
                "Vectors in 2-Space & 3-Space",
                "Complex Numbers"
            ]},
        }

        # GUI - Header
        tk.Label(
            self,
            text="Study Assistant Suite",
            font=("Arial", 22, "bold"),
            bg=backgroundColour,
            fg=accentColour
        ).pack(pady=(30, 10))
        
        tk.Label(
            self,
            text="Select a module to start studying",
            font=("Arial", 11),
            bg=backgroundColour,
            fg=foregroundColour
        ).pack(pady=(0, 20))

        # Grid container for module buttons
        gridContainer = tk.Frame(self, bg=backgroundColour)
        gridContainer.pack(padx=40, pady=10)
        
        row_idx = 0
        col_idx = 0

        for code, info in self.courseInformation.items():
            displayText = f"{code}\n{info['name']}"
            
            buttonTopic = tk.Button(
                gridContainer,
                text=displayText,
                command=lambda c=code, n=info['name'], ch=info['chapters']: self.openModuleWindow(c, n, ch),
                width=22,
                bg=buttonBg,
                fg=foregroundColour,
                font=("Arial", 11, "bold"),
                relief='flat',
                cursor='hand2',
                pady=15
            )
            
            # Add hover effects
            buttonTopic.bind("<Enter>", lambda e: e.widget.config(bg=buttonHover))
            buttonTopic.bind("<Leave>", lambda e: e.widget.config(bg=buttonBg))
            
            buttonTopic.grid(row=row_idx, column=col_idx, padx=12, pady=12)
            
            # Grid navigation logic
            col_idx += 1                          # Move to the next column
            if col_idx >= 2:                      # Check if we have placed 2 buttons
                col_idx = 0                       # Reset column back to 0
                row_idx += 1                      # Move to the next row

    def openModuleWindow(self, code, name, chapters):
        moduleWindow(self, code, name, chapters)


if __name__ == "__main__":
    app = pocketStudentSuite()                     
    app.mainloop()