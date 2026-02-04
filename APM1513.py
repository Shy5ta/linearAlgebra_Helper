import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

#+++++++++++++++ Color Scheme +++++++++++++++++++
backgroundColour = '#282c34'
foregroundColour = '#c6c6c6'
accentColour = '#61afef'
buttonBg = '#3c424a'
buttonHover = '#4a5260'

# Configure ttk styles for better appearance
style = ttk.Style()
style.theme_use('default')
style.configure('TNotebook', background=backgroundColour, borderwidth=0)
style.configure('TNotebook.Tab', background=buttonBg, foreground=foregroundColour, 
                padding=[10, 5], font=('Arial', 10))
style.map('TNotebook.Tab', background=[('selected', accentColour)], 
          foreground=[('selected', 'white')])

# Constants for problem types
PROBLEM_DETERMINANT = "Calculate Determinant"
PROBLEM_INVERSE = "Find Matrix Inverse"
PROBLEM_TRACE = "Calculate Trace"
PROBLEM_TRANSPOSE = "Transpose Matrix"
PROBLEM_DIRECT_METHOD = "Direct Method (Ax=b)"
PROBLEM_GAUSS_SEIDEL = "Iterative (Gauss-Seidel)"
PROBLEM_EIGENVALUES = "Calculate Eigenvalues"
PROBLEM_LEAST_SQUARES = "Method of Least Squares"
PROBLEM_NULL_SPACE = "Find General Solution"
PROBLEM_LINEAR_PROG = "Solve LP (glpk)"

#+++++++++++++++ APM1513 Problem Solver Window +++++++++++++++++++
class APM1513Window(tk.Toplevel):
    def __init__(self, parent, topicName):
        # This initializes the problem solver window
        super().__init__(parent)
        
        self.title(f"APM1513 - {topicName}")
        self.geometry("1100x700")
        self.configure(bg=backgroundColour)
        
        # Disable the module window while this is open
        parent.attributes('-disabled', True)
        
        # Protocol to run when the user closes the window
        self.protocol("WM_DELETE_WINDOW", lambda: self.on_close(parent))
        
        # Store the topic name for reference
        self.topicName = topicName
        
        # Create the appropriate interface based on topic
        self.createInterface()
    
    def createInterface(self):
        # 1. TOP HEADER FRAME (Title + Dropdown)
        topFrame = tk.Frame(self, bg=backgroundColour)
        topFrame.pack(side="top", fill="x", padx=20, pady=(15, 10))

        # Title
        tk.Label(
            topFrame, 
            text=f"📐 {self.topicName}", 
            font=("Arial", 16, "bold"), 
            bg=backgroundColour, 
            fg=accentColour
        ).pack(side="top", anchor="center")

        # Dropdown Selection Area
        selectionFrame = tk.Frame(topFrame, bg=backgroundColour)
        selectionFrame.pack(side="top", pady=(10, 0))

        tk.Label(
            selectionFrame,
            text="Problem Type:",
            font=("Arial", 11, "bold"),
            bg=backgroundColour,
            fg=foregroundColour
        ).pack(side="left", padx=(0, 10))
        
        problemTypes = self.getProblemTypes()
        
        # FIX: master=self ensures the variable connects to this window
        self.problemTypeVar = tk.StringVar(master=self)
        self.problemTypeVar.set(problemTypes[0])
        
        problemDropdown = ttk.Combobox(
            selectionFrame,
            textvariable=self.problemTypeVar,
            values=problemTypes,
            state="readonly",
            width=35,
            font=("Arial", 10)
        )
        problemDropdown.pack(side="left")
        problemDropdown.bind("<<ComboboxSelected>>", self.onProblemTypeChanged)

        # 2. MAIN CONTENT FRAME (Side-by-Side Layout)
        contentFrame = tk.Frame(self, bg=backgroundColour)
        contentFrame.pack(fill="both", expand=True, padx=20, pady=10)

        # --- LEFT SIDE: INPUT ---
        leftFrame = tk.Frame(contentFrame, bg=backgroundColour)
        leftFrame.pack(side="left", fill="both", expand=False, padx=(0, 10))

        self.hintLabel = tk.Label(
            leftFrame,
            text="Enter Matrix / Data:",
            font=("Arial", 10, "bold"),
            bg=backgroundColour,
            fg=foregroundColour,
            anchor="w"
        )
        self.hintLabel.pack(fill="x", pady=(0, 5))

        self.inputText = scrolledtext.ScrolledText(
            leftFrame,
            width= 40,
            font=("Courier", 11),
            bg='#1e1e1e',
            fg=foregroundColour,
            insertbackground=accentColour,
            wrap="word",
            relief="flat",
            borderwidth=1
        )
        self.inputText.pack(fill="both", expand=True)

        # --- RIGHT SIDE: OUTPUT ---
        rightFrame = tk.Frame(contentFrame, bg=backgroundColour)
        rightFrame.pack(side="right", fill="both", expand=False, padx=(10, 0))

        tk.Label(
            rightFrame, 
            text="Generated Octave Code:", 
            font=("Arial", 10, "bold"), 
            bg=backgroundColour, 
            fg=foregroundColour, 
            anchor="w"
        ).pack(fill="x", pady=(0, 5))

        self.resultText = scrolledtext.ScrolledText(
            rightFrame, 
            font=("Courier", 11),
            bg='#1e1e1e', 
            fg='#98c379', # Green text
            wrap="word",
            relief="flat",
            borderwidth=1
        )
        self.resultText.pack(fill="both", expand=True)

        # 3. BOTTOM BUTTON BAR
        buttonFrame = tk.Frame(self, bg=backgroundColour)
        buttonFrame.pack(side="bottom", fill="x", padx=20, pady=20, before=contentFrame) #adding before fixes the bug on linear 
                                                                                        #programming which hides the buttons

        # Generate Button
        tk.Button(
            buttonFrame,
            text="📝 Generate Code",
            command=self.generateSolution,
            bg=accentColour,
            fg='white',
            font=("Arial", 11, "bold"),
            relief='flat',
            cursor='hand2',
            width=18,
            pady=8
        ).pack(side="left", padx=(0, 10))

        # Clear Button
        tk.Button(
            buttonFrame,
            text="🗑️ Clear All",
            command=self.clearInput,
            bg=buttonBg,
            fg=foregroundColour,
            font=("Arial", 11),
            relief='flat',
            cursor='hand2',
            width=12,
            pady=8
        ).pack(side="left", padx=10)

        # NEW: Back Button (Replaces the need to press X)
        tk.Button(
            buttonFrame,
            text="← Back to Menu",
            command=lambda: self.on_close(self.master), # Calls your existing close logic
            bg='#e06c75', # Red color
            fg='white',
            font=("Arial", 11, "bold"),
            relief='flat',
            cursor='hand2',
            width=15,
            pady=8
        ).pack(side="right")

        # Initialize hints
        self.updateInputHint()
    
    def getProblemTypes(self):
        # Returns a list of problem types based on the selected topic
        problemMap = {
            "Matrix Properties and Manipulation": [
                PROBLEM_DETERMINANT,
                PROBLEM_INVERSE,
                PROBLEM_TRACE,
                PROBLEM_TRANSPOSE
            ],
            "Solving Square Linear Systems": [
                PROBLEM_DIRECT_METHOD,
                PROBLEM_GAUSS_SEIDEL
            ],
            "Eigenvalues and Eigenvectors": [
                PROBLEM_EIGENVALUES
            ],
            "Overdetermined Systems and Least Squares": [
                PROBLEM_LEAST_SQUARES
            ],
            "Underdetermined Systems and Null Space": [
                PROBLEM_NULL_SPACE
            ],
            "Linear Programming": [
                PROBLEM_LINEAR_PROG
            ]
        }
        return problemMap.get(self.topicName, ["Standard Operation"])
    
    def onProblemTypeChanged(self, event=None):
        """Handle problem type dropdown change"""
        # Clear inputs when switching problem types
        self.resultText.delete("1.0", "end")
        # Update hint
        self.updateInputHint()
    
    def updateInputHint(self, event=None):
        """Update the label above the input box with instructions"""
        selected = self.problemTypeVar.get()
        
        hint_text = "Enter Matrix / Data:"
        
        if selected == PROBLEM_DIRECT_METHOD:
            hint_text = "Enter Matrix A (rows), leave a blank line, then Enter Vector b.\nExample:\n2 1\n1 3\n\n5\n8"
        elif selected == PROBLEM_GAUSS_SEIDEL:
            hint_text = "Enter Matrix A (rows), leave a blank line, then Enter Vector b.\nExample:\n4 1\n1 3\n\n7\n8"
        elif selected == PROBLEM_LINEAR_PROG:
            hint_text = "Enter c, leave blank line, Enter A, leave blank line, Enter b.\nExample (Maximize):\n40 60\n\n2 1\n1 1\n\n70\n40"
        elif selected == PROBLEM_LEAST_SQUARES:
            hint_text = "Enter Matrix A (rows), leave a blank line, then Enter Vector b.\nExample:\n1 2\n2 3\n3 5\n\n1\n2\n3"
        elif selected == PROBLEM_NULL_SPACE:
            hint_text = "Enter Matrix A (underdetermined system).\nExample:\n1 2 3\n4 5 6"
        else:
            hint_text = "Enter Matrix (space separated values, new line for each row).\nExample:\n1 2 3\n4 5 6\n7 8 9"
            
        self.hintLabel.config(text=hint_text)

    def clearInput(self):
        """Clear both input and result text boxes"""
        self.inputText.delete("1.0", "end")
        self.resultText.delete("1.0", "end")
    
    def parse_matrix(self, text):
        """Helper to parse matrix text into list of lists"""
        matrix = []
        rows = text.strip().split('\n')
        for i, row in enumerate(rows, 1):
            if not row.strip() or row.strip().startswith("#"):
                continue
            try:
                parsed_row = [float(x) for x in row.split()]
                if not parsed_row:
                    raise ValueError(f"Empty row at line {i}")
                matrix.append(parsed_row)
            except ValueError as e:
                raise ValueError(f"Invalid data at line {i}: {row.strip()}\nError: {str(e)}")
        
        if not matrix:
            raise ValueError("No valid matrix data found")
        
        # Check for consistent row lengths
        row_length = len(matrix[0])
        for i, row in enumerate(matrix, 1):
            if len(row) != row_length:
                raise ValueError(f"Inconsistent row length at row {i}: expected {row_length}, got {len(row)}")
        
        return matrix

    def validate_square_matrix(self, matrix, operation_name):
        """Validate that a matrix is square"""
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0
        if rows != cols:
            raise ValueError(f"{operation_name} requires a square matrix. Got {rows}x{cols} matrix.")
        return True

    def format_octave_matrix(self, matrix):
        """Converts [[1,2],[3,4]] to '[1 2; 3 4]' string for Octave"""
        row_strs = []
        for row in matrix:
            row_strs.append(" ".join(str(x) for x in row))
        return "[" + "; ".join(row_strs) + "]"

    def generateSolution(self):
        """Main logic to generate Octave Code strings"""
        input_str = self.inputText.get("1.0", "end")
        problem = self.problemTypeVar.get()
        octave_code = ""

        try:
            # Clear previous results
            self.resultText.delete("1.0", "end")
            
            # Validate input
            if not input_str.strip():
                raise ValueError("Input is empty. Please enter the required data.")
            
            # Parse based on problem type
            if problem == PROBLEM_DIRECT_METHOD:
                parts = input_str.split('\n\n')
                if len(parts) < 2:
                    raise ValueError("Separate Matrix A and Vector b with a blank line.")
                
                matrix_A = self.parse_matrix(parts[0])
                vector_b = self.parse_matrix(parts[1])
                
                self.validate_square_matrix(matrix_A, "Direct Method")
                
                if len(vector_b) != len(matrix_A):
                    raise ValueError(f"Vector b must have {len(matrix_A)} rows to match Matrix A.")
                
                str_A = self.format_octave_matrix(matrix_A)
                str_b = self.format_octave_matrix(vector_b)
                
                octave_code = f"""% Octave Script: Solve Ax = b using Direct Method
A = {str_A};
b = {str_b};

% Solve using left division operator
x = A \\ b;

disp("Solution vector x:");
disp(x);
"""

            elif problem == PROBLEM_GAUSS_SEIDEL:
                parts = input_str.split('\n\n')
                if len(parts) < 2:
                    raise ValueError("Separate Matrix A and Vector b with a blank line.")
                
                matrix_A = self.parse_matrix(parts[0])
                vector_b = self.parse_matrix(parts[1])
                
                self.validate_square_matrix(matrix_A, "Gauss-Seidel Method")
                
                if len(vector_b) != len(matrix_A):
                    raise ValueError(f"Vector b must have {len(matrix_A)} rows to match Matrix A.")
                
                str_A = self.format_octave_matrix(matrix_A)
                str_b = self.format_octave_matrix(vector_b)
                
                octave_code = f"""% Octave Script: Gauss-Seidel Iterative Method
A = {str_A};
b = {str_b};

% Gauss-Seidel function
function xnew = gauss_seidel(A, b, xold)
    n = size(A)(1);
    At = A;
    xnew = xold;
    for k = 1:n
        At(k,k) = 0;
    end
    for k = 1:n
        xnew(k) = (b(k) - At(k,:)*xnew) / A(k,k);
    end
endfunction

% Parameters
max_iter = 100;
tol = 1e-6;
n = length(b);
xold = zeros(n, 1);  % Initial guess

% Iterative solution
for iter = 1:max_iter
    xnew = gauss_seidel(A, b, xold);
    
    % Check convergence
    if norm(xnew - xold, inf) < tol
        fprintf('Converged in %d iterations\\n', iter);
        break;
    end
    
    xold = xnew;
end

disp("Solution vector x:");
disp(xnew);
"""

            elif problem == PROBLEM_LINEAR_PROG:
                parts = input_str.split('\n\n')
                if len(parts) < 3:
                    raise ValueError("Separate c, A, and b with blank lines (need 3 sections).")
                
                matrix_c = self.parse_matrix(parts[0])
                matrix_A = self.parse_matrix(parts[1])
                matrix_b = self.parse_matrix(parts[2])
                
                # Validate dimensions
                n_vars = len(matrix_c[0])  # Number of variables from c
                
                if len(matrix_A[0]) != n_vars:
                    raise ValueError(f"Matrix A must have {n_vars} columns to match c vector.")
                
                if len(matrix_b) != len(matrix_A):
                    raise ValueError(f"Vector b must have {len(matrix_A)} rows to match Matrix A.")
                
                str_c = self.format_octave_matrix(matrix_c)
                str_A = self.format_octave_matrix(matrix_A)
                str_b = self.format_octave_matrix(matrix_b)
                
                # Create bounds vector dynamically
                lb_str = "zeros(" + str(n_vars) + ", 1)"
                ctype_str = '"' + "U" * len(matrix_A) + '"'
                vartype_str = '"' + "C" * n_vars + '"'
                
                octave_code = f"""% Octave Script: Linear Programming (Simplex)
c = {str_c}';  % Objective function coefficients
A = {str_A};   % Constraint matrix
b = {str_b};   % Constraint limits

% Standard bounds (x >= 0)
lb = {lb_str}; 
ub = []; 
ctype = {ctype_str}; % Upper bounds constraints
vartype = {vartype_str}; % Continuous variables
s = -1; % Maximize (-1) or Minimize (1)

[xmax, fmax] = glpk(c, A, b, lb, ub, ctype, vartype, s);

disp("Optimal x:");
disp(xmax);
disp("Maximum Value:");
disp(fmax);
"""

            elif problem == PROBLEM_LEAST_SQUARES:
                parts = input_str.split('\n\n')
                if len(parts) < 2:
                    raise ValueError("Separate Matrix A and Vector b with a blank line.")
                
                matrix_A = self.parse_matrix(parts[0])
                vector_b = self.parse_matrix(parts[1])
                
                if len(vector_b) != len(matrix_A):
                    raise ValueError(f"Vector b must have {len(matrix_A)} rows to match Matrix A.")
                
                str_A = self.format_octave_matrix(matrix_A)
                str_b = self.format_octave_matrix(vector_b)
                
                octave_code = f"""% Least Squares Solution (Overdetermined System)
A = {str_A};
b = {str_b};

% Compute least squares solution
x = (A' * A) \\ (A' * b);

disp("Least Squares Solution x:");
disp(x);

% Calculate residual
residual = b - A * x;
disp("Residual norm:");
disp(norm(residual));
"""

            elif problem == PROBLEM_NULL_SPACE:
                matrix_A = self.parse_matrix(input_str)
                str_A = self.format_octave_matrix(matrix_A)
                
                octave_code = f"""% Find General Solution (Null Space)
A = {str_A};

% Find null space
N = null(A);

disp("Null space basis vectors:");
disp(N);

% General solution: x = x_particular + linear combination of null space
disp("General solution: x = x_p + c1*N(:,1) + c2*N(:,2) + ...");
"""

            else:
                # Standard single matrix operations
                matrix = self.parse_matrix(input_str)
                str_A = self.format_octave_matrix(matrix)
                
                if problem == PROBLEM_DETERMINANT:
                    self.validate_square_matrix(matrix, "Determinant calculation")
                    octave_code = f"""% Calculate Determinant
A = {str_A};
d = det(A);
disp("Determinant:");
disp(d);
"""
                elif problem == PROBLEM_INVERSE:
                    self.validate_square_matrix(matrix, "Matrix inverse")
                    octave_code = f"""% Calculate Inverse
A = {str_A};
d = det(A);
if abs(d) < 1e-10
    disp("Matrix is singular (det ≈ 0), no inverse exists.");
else
    invA = inv(A);
    disp("Inverse Matrix:");
    disp(invA);
end
"""
                elif problem == PROBLEM_TRACE:
                    self.validate_square_matrix(matrix, "Trace calculation")
                    octave_code = f"""% Calculate Trace
A = {str_A};
t = trace(A);
disp("Trace:");
disp(t);
"""
                elif problem == PROBLEM_TRANSPOSE:
                    octave_code = f"""% Transpose Matrix
A = {str_A};
AT = A';
disp("Transposed Matrix:");
disp(AT);
"""
                elif problem == PROBLEM_EIGENVALUES:
                    self.validate_square_matrix(matrix, "Eigenvalue calculation")
                    octave_code = f"""% Eigenvalues and Eigenvectors
A = {str_A};
[V, D] = eig(A);
disp("Eigenvalues (Diagonal of D):");
disp(diag(D));
disp("Eigenvectors (Columns of V):");
disp(V);
"""
                else:
                    raise ValueError(f"Code generation for '{problem}' is not yet implemented.")

            self.resultText.insert("end", octave_code)

        except ValueError as e:
            octave_code = f"% Input Error: {str(e)}"
            self.resultText.insert("1.0", octave_code)
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            octave_code = f"% Unexpected Error: {str(e)}"
            self.resultText.insert("1.0", octave_code)
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
    
    def on_close(self, parent):
        """Re-enable the parent window and close this window"""
        parent.attributes('-disabled', False)
        self.destroy()

    def on_close(self, parent):
        parent.attributes('-disabled', False)
        self.destroy()