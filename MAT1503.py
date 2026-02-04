import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import math
import cmath

#+++++++++++++++ Color Scheme +++++++++++++++++++
backgroundColour = '#282c34'
foregroundColour = '#c6c6c6'
accentColour = '#61afef'
buttonBg = '#3c424a'
buttonHover = '#4a5260'

#+++++++++++++++ MAT1503 Solver Window +++++++++++++++++++
class MAT1503Window(tk.Toplevel):
    def __init__(self, parent, topicName):
        super().__init__(parent)
        
        self.title(f"MAT1503 - {topicName}")
        self.geometry("950x750")
        self.configure(bg=backgroundColour)
        
        # Disable parent interaction
        parent.attributes('-disabled', True)
        self.protocol("WM_DELETE_WINDOW", lambda: self.on_close(parent))
        
        self.topicName = topicName
        self.entries = {} 
        
        self.createInterface()

    def createInterface(self):
        # --- Header ---
        headerFrame = tk.Frame(self, bg=backgroundColour)
        headerFrame.pack(pady=20, padx=20, fill="x")
        
        tk.Label(
            headerFrame,
            text=f"📐 {self.topicName}",
            font=("Arial", 18, "bold"),
            bg=backgroundColour,
            fg=accentColour
        ).pack()
        
        tk.Label(
            headerFrame,
            text="Enter parameters for Step-by-Step Linear Algebra Solutions",
            font=("Arial", 10),
            bg=backgroundColour,
            fg=foregroundColour
        ).pack(pady=(5, 0))

        # --- Main Layout ---
        contentFrame = tk.Frame(self, bg=backgroundColour)
        contentFrame.pack(fill="both", expand=True, padx=20, pady=10)

        # 1. LEFT SIDE: Inputs
        leftPanel = tk.Frame(contentFrame, bg=backgroundColour, width=320)
        leftPanel.pack(side="left", fill="y", padx=(0, 20), anchor="n")

        # Task Selector
        tk.Label(
            leftPanel, 
            text="Select Task:", 
            bg=backgroundColour, 
            fg=foregroundColour,
            font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))
        
        self.taskVar = tk.StringVar()
        tasks = self.getTasks()
        self.taskVar.set(tasks[0])
        
        dropdown = ttk.Combobox(leftPanel, textvariable=self.taskVar, values=tasks, state="readonly", width=35)
        dropdown.pack(fill="x", pady=(0, 15))
        dropdown.bind("<<ComboboxSelected>>", self.updateInputFields)

        # Input Fields Container
        self.inputFrame = tk.LabelFrame(
            leftPanel, 
            text="Input Data", 
            bg=backgroundColour, 
            fg=accentColour, 
            font=("Arial", 10, "bold"),
            padx=10, pady=10
        )
        self.inputFrame.pack(fill="x", pady=5)
        
        # Initialize inputs
        self.updateInputFields()

        # Calculate Button
        solveButton = tk.Button(
            leftPanel,
            text="📝 Solve Step-by-Step",
            command=self.solve_step_by_step,
            bg=accentColour,
            fg='white',
            font=("Arial", 11, "bold"),
            relief='flat',
            cursor='hand2',
            pady=10
        )
        solveButton.pack(fill="x", pady=20)

        # Clear Button
        clearButton = tk.Button(
            leftPanel,
            text="🗑️ Clear",
            command=self.clearInputs,
            bg=buttonBg,
            fg=foregroundColour,
            font=("Arial", 10),
            relief='flat',
            cursor='hand2'
        )
        clearButton.pack(fill="x")

        # 2. RIGHT SIDE: Solution Display
        rightPanel = tk.Frame(contentFrame, bg=backgroundColour)
        rightPanel.pack(side="right", fill="both", expand=True)

        tk.Label(
            rightPanel, 
            text="Mathematical Solution:", 
            bg=backgroundColour, 
            fg=foregroundColour,
            font=("Arial", 10, "bold")
        ).pack(anchor="w")

        self.solutionText = scrolledtext.ScrolledText(
            rightPanel,
            font=("Courier", 11),
            bg='#1e1e1e',
            fg=foregroundColour,
            wrap="word",
            padx=15, pady=15
        )
        self.solutionText.pack(fill="both", expand=True, pady=5)
        
        # Back Button
        tk.Button(
            self,
            text="← Back to Topics",
            command=lambda: self.on_close(self.master),
            bg=backgroundColour,
            fg=accentColour,
            font=("Arial", 10),
            relief='flat',
            cursor='hand2',
            pady=10
        ).pack(side="bottom", pady=10)

    def getTasks(self):
        # Return tasks based on the Topic Name (Chapters)
        if "System" in self.topicName or "Matrices" in self.topicName:
            return ["Solve 2x2 System (Cramer's Rule)", "Matrix Multiplication (2x2)", "Inverse of 2x2 Matrix"]
        elif "Determinant" in self.topicName:
            return ["Determinant (2x2)", "Determinant (3x3)"]
        elif "Vectors" in self.topicName:
            return ["Dot Product", "Cross Product", "Angle Between Vectors", "Projection of u onto v"]
        elif "Complex" in self.topicName:
            return ["Complex Arithmetic (+, -, *, /)", "Convert to Polar Form", "De Moivre's Theorem (Powers)"]
        else:
            return ["Standard Calculation"]

    def updateInputFields(self, event=None):
        for widget in self.inputFrame.winfo_children():
            widget.destroy()
        self.entries = {}
        
        task = self.taskVar.get()
        fields = []
        
        # --- SYSTEMS & MATRICES ---
        if "Solve 2x2 System" in task:
            tk.Label(self.inputFrame, text="System: ax + by = e, cx + dy = f", bg=backgroundColour, fg=foregroundColour).pack()
            fields = [("a", "2"), ("b", "3"), ("e", "5"), ("c", "4"), ("d", "1"), ("f", "2")]
        elif "Matrix Multiplication" in task:
            tk.Label(self.inputFrame, text="Matrix A (2x2) * Matrix B (2x2)", bg=backgroundColour, fg=foregroundColour).pack()
            fields = [("A11", "1"), ("A12", "2"), ("A21", "3"), ("A22", "4"), 
                      ("B11", "2"), ("B12", "0"), ("B21", "1"), ("B22", "2")]
        elif "Inverse" in task:
            fields = [("A11", "4"), ("A12", "7"), ("A21", "2"), ("A22", "6")]

        # --- DETERMINANTS ---
        elif "Determinant (2x2)" in task:
            fields = [("a11", "2"), ("a12", "3"), ("a21", "4"), ("a22", "5")]
        elif "Determinant (3x3)" in task:
            fields = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "0"), ("e", "1"), ("f", "4"), ("g", "5"), ("h", "6"), ("i", "0")]

        # --- VECTORS ---
        elif "Dot Product" in task or "Angle" in task or "Projection" in task:
            fields = [("u1", "1"), ("u2", "2"), ("u3", "3"), ("v1", "4"), ("v2", "-5"), ("v3", "6")]
        elif "Cross Product" in task:
            fields = [("u1", "1"), ("u2", "0"), ("u3", "1"), ("v1", "2"), ("v2", "3"), ("v3", "0")]

        # --- COMPLEX NUMBERS ---
        elif "Complex Arithmetic" in task:
            fields = [("Real z1", "3"), ("Imag z1", "2"), ("Real z2", "1"), ("Imag z2", "-4")]
        elif "Polar Form" in task:
            fields = [("Real Part (a)", "1"), ("Imag Part (b)", "1")]
        elif "De Moivre" in task:
            fields = [("Real Part", "1"), ("Imag Part", "1"), ("Power n", "5")]

        # Create Entry Boxes
        for text, default in fields:
            row = tk.Frame(self.inputFrame, bg=backgroundColour)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=text, width=15, anchor="w", bg=backgroundColour, fg=accentColour, font=("Arial", 9, "bold")).pack(side="left")
            ent = tk.Entry(row, bg=buttonBg, fg="white", insertbackground=accentColour, relief="flat", font=("Arial", 10))
            ent.insert(0, default)
            ent.pack(side="right", fill="x", expand=True)
            self.entries[text] = ent

    def solve_step_by_step(self):
        task = self.taskVar.get()
        steps = ""
        
        try:
            data = {k: float(v.get()) for k, v in self.entries.items()}
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")
            return

        # ================= MATRICES & SYSTEMS =================
        if "Solve 2x2 System" in task:
            a, b, e = data["a"], data["b"], data["e"]
            c, d, f = data["c"], data["d"], data["f"]
            
            detA = a*d - b*c
            dx = e*d - b*f
            dy = a*f - e*c
            
            steps += "TOPIC: CRAMER'S RULE (2x2)\n----------------------------\n"
            steps += f"System:\n   {a}x + {b}y = {e}\n   {c}x + {d}y = {f}\n\n"
            steps += "1. Calculate Determinant D:\n"
            steps += f"   D = (a*d) - (b*c) = ({a}*{d}) - ({b}*{c}) = {detA}\n\n"
            
            if detA == 0:
                steps += "   Since D = 0, the system has no unique solution.\n"
            else:
                steps += "2. Calculate Dx (Replace x-column with constants):\n"
                steps += f"   Dx = (e*d) - (b*f) = ({e}*{d}) - ({b}*{f}) = {dx}\n\n"
                steps += "3. Calculate Dy (Replace y-column with constants):\n"
                steps += f"   Dy = (a*f) - (e*c) = ({a}*{f}) - ({e}*{c}) = {dy}\n\n"
                steps += "4. Solve for x and y:\n"
                steps += f"   x = Dx / D = {dx} / {detA} = {dx/detA:.2f}\n"
                steps += f"   y = Dy / D = {dy} / {detA} = {dy/detA:.2f}\n"

        elif "Inverse" in task:
            a, b, c, d = data["A11"], data["A12"], data["A21"], data["A22"]
            det = a*d - b*c
            
            steps += "TOPIC: INVERSE MATRIX (2x2)\n---------------------------\n"
            steps += f"Matrix A = [[{a}, {b}], [{c}, {d}]]\n\n"
            steps += "1. Calculate Determinant:\n"
            steps += f"   det(A) = ad - bc = ({a})({d}) - ({b})({c}) = {det}\n\n"
            
            if det == 0:
                steps += "   Since det(A) = 0, the matrix is Singular and has NO Inverse.\n"
            else:
                steps += "2. Swap main diagonal, change signs of off-diagonal:\n"
                steps += f"   Adjoint = [[{d}, {-b}], [{-c}, {a}]]\n\n"
                steps += "3. Multiply by 1/det(A):\n"
                steps += f"   A^(-1) = (1/{det}) * Adjoint\n"
                steps += f"   A^(-1) = [[{d/det:.2f}, {-b/det:.2f}], [{-c/det:.2f}, {a/det:.2f}]]\n"

        # ================= VECTORS =================
        elif "Dot Product" in task:
            u = [data["u1"], data["u2"], data["u3"]]
            v = [data["v1"], data["v2"], data["v3"]]
            dot_prod = sum(i*j for i, j in zip(u, v))
            
            steps += "TOPIC: DOT PRODUCT\n------------------\n"
            steps += f"Vectors:\n   u = {u}\n   v = {v}\n\n"
            steps += "Formula: u.v = u1*v1 + u2*v2 + u3*v3\n"
            steps += f"   = ({u[0]}*{v[0]}) + ({u[1]}*{v[1]}) + ({u[2]}*{v[2]})\n"
            steps += f"   = {u[0]*v[0]} + {u[1]*v[1]} + {u[2]*v[2]}\n"
            steps += f"   = {dot_prod}\n"

        elif "Cross Product" in task:
            u1, u2, u3 = data["u1"], data["u2"], data["u3"]
            v1, v2, v3 = data["v1"], data["v2"], data["v3"]
            
            cx = u2*v3 - u3*v2
            cy = u3*v1 - u1*v3
            cz = u1*v2 - u2*v1
            
            steps += "TOPIC: CROSS PRODUCT\n--------------------\n"
            steps += f"u = <{u1}, {u2}, {u3}>, v = <{v1}, {v2}, {v3}>\n\n"
            steps += "Formula (Determinant method):\n"
            steps += "   i(u2v3 - u3v2) - j(u1v3 - u3v1) + k(u1v2 - u2v1)\n\n"
            steps += f"1. i-component: ({u2}*{v3}) - ({u3}*{v2}) = {cx}\n"
            steps += f"2. j-component: -[({u1}*{v3}) - ({u3}*{v1})] = -[{u1*v3 - u3*v1}] = {cy}\n"
            steps += f"3. k-component: ({u1}*{v2}) - ({u2}*{v1}) = {cz}\n\n"
            steps += f"Result: u x v = <{cx}, {cy}, {cz}>\n"

        # ================= COMPLEX NUMBERS =================
        elif "Polar Form" in task:
            a, b = data["Real Part (a)"], data["Imag Part (b)"]
            r = math.sqrt(a**2 + b**2)
            theta = math.atan2(b, a)
            deg = math.degrees(theta)
            
            steps += "TOPIC: COMPLEX POLAR FORM\n---------------------------\n"
            steps += f"z = {a} + {b}i\n\n"
            steps += "1. Calculate Modulus |z| (r):\n"
            steps += f"   r = sqrt(a^2 + b^2) = sqrt({a}^2 + {b}^2)\n"
            steps += f"   r = sqrt({a**2 + b**2}) = {r:.4f}\n\n"
            steps += "2. Calculate Argument (theta):\n"
            steps += f"   theta = atan(b/a) = atan({b}/{a})\n"
            steps += f"   theta = {theta:.4f} radians ({deg:.2f} degrees)\n\n"
            steps += "3. Polar Form:\n"
            steps += f"   z = {r:.2f}(cos({deg:.2f}°) + i*sin({deg:.2f}°))\n"

        elif "De Moivre" in task:
            a, b, n = data["Real Part"], data["Imag Part"], int(data["Power n"])
            r = math.sqrt(a**2 + b**2)
            theta = math.atan2(b, a)
            
            steps += "TOPIC: DE MOIVRE'S THEOREM\n--------------------------\n"
            steps += f"Calculate ({a} + {b}i)^{n}\n\n"
            steps += "1. Convert to Polar:\n"
            steps += f"   r = {r:.2f}, theta = {math.degrees(theta):.2f}°\n\n"
            steps += "2. Apply Theorem: z^n = r^n * (cos(n*theta) + i*sin(n*theta))\n"
            steps += f"   r^{n} = {r}^{n} = {r**n:.2f}\n"
            steps += f"   angle = {n} * {math.degrees(theta):.2f}° = {n*math.degrees(theta):.2f}°\n\n"
            
            new_r = r**n
            new_theta = n * theta
            final_real = new_r * math.cos(new_theta)
            final_imag = new_r * math.sin(new_theta)
            
            steps += "3. Convert back to Rectangular:\n"
            steps += f"   Real = {new_r:.2f} * cos({math.degrees(new_theta):.2f}°) = {final_real:.2f}\n"
            steps += f"   Imag = {new_r:.2f} * sin({math.degrees(new_theta):.2f}°) = {final_imag:.2f}\n\n"
            steps += f"Result: {final_real:.2f} + {final_imag:.2f}i\n"

        else:
            steps = "Solution logic for this task is coming soon."

        self.solutionText.delete("1.0", "end")
        self.solutionText.insert("1.0", steps)

    def clearInputs(self):
        self.updateInputFields()
        self.solutionText.delete("1.0", "end")

    def on_close(self, parent):
        parent.attributes('-disabled', False)
        self.destroy()