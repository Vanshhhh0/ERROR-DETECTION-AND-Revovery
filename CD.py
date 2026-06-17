from __future__ import annotations
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import ply.lex as lex
import ply.yacc as yacc
from dataclasses import dataclass
from typing import List, Dict

# -----------------------------
# Mini Language Compiler
# -----------------------------
# This file implements a simple compiler for a mini-language with:
# - Lexer (tokenizer)
# - Parser (with semantic analysis and error recovery)
# - AST and TAC (Three Address Code) generation
# - Simple interpreter for TAC
# - GUI for interactive use
#
# Author: <Your Name>
# Date: <Date>
# -----------------------------

# -----------------------------
# LEXER
# -----------------------------
class MiniLexer:
    """
    Lexical analyzer for the mini-language.
    Tokenizes input source code into a stream of tokens.
    """
    tokens = (
        'ID', 'NUMBER',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
        'ASSIGN',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'SEMI'
    )

    reserved = {'if': 'IF', 'while': 'WHILE'}
    tokens += tuple(reserved.values())

    # Token regex patterns
    t_PLUS   = r'\+'
    t_MINUS  = r'-'
    t_TIMES  = r'\*'
    t_DIVIDE = r'/'
    t_ASSIGN = r'='
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_SEMI   = r';'
    t_ignore = ' \t\r'

    def t_COMMENT(self, t):
        r'//.*'
        # Ignore comments
        pass

    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_ID(self, t):
        r'[A-Za-z_][A-Za-z0-9_]*'
        t.type = self.reserved.get(t.value, 'ID')
        return t

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    def t_error(self, t):
        self.errors.append(f"Illegal character '{t.value[0]}' at line {t.lineno}")
        t.lexer.skip(1)

    def __init__(self):
        self.lexer  = lex.lex(module=self, debug=False)
        self.errors: List[str] = []

    def scan(self, text: str):
        """
        Scan the input text and return a list of tokens and any errors found.
        """
        self.lexer.lineno = 1
        self.errors.clear()
        self.lexer.input(text)
        out = []
        while (tok := self.lexer.token()):
            out.append((tok.type, tok.value, tok.lineno, tok.lexpos))
        return out, self.errors.copy()

# -----------------------------
# AST NODES + TAC EMISSION
# -----------------------------
_tmp_counter = 0

def new_tmp() -> str:
    """Generate a new temporary variable name for TAC."""
    global _tmp_counter
    _tmp_counter += 1
    return f"t{_tmp_counter}"

@dataclass
class Node:
    """Base class for all AST nodes."""
    def gen(self, tac: List[str]):
        raise NotImplementedError

@dataclass
class Number(Node):
    val: int
    def gen(self, tac):
        tmp = new_tmp()
        tac.append(f"{tmp} = {self.val}")
        return tmp

@dataclass
class Identifier(Node):
    name: str
    def gen(self, tac):
        return self.name

@dataclass
class BinOp(Node):
    op: str
    left: Node
    right: Node
    def gen(self, tac):
        l = self.left.gen(tac)
        r = self.right.gen(tac)
        tmp = new_tmp()
        tac.append(f"{tmp} = {l} {self.op} {r}")
        return tmp

@dataclass
class Assign(Node):
    name: str
    expr: Node
    def gen(self, tac):
        et = self.expr.gen(tac)
        tac.append(f"{self.name} = {et}")

@dataclass
class Seq(Node):
    stmts: List[Node]
    def gen(self, tac):
        for s in self.stmts:
            s.gen(tac)

@dataclass
class If(Node):
    cond: Node
    body: Seq
    def gen(self, tac):
        c = self.cond.gen(tac)
        lbl = new_tmp()
        tac.append(f"ifFalse {c} goto {lbl}")
        self.body.gen(tac)
        tac.append(f"label {lbl}")

@dataclass
class While(Node):
    cond: Node
    body: Seq
    def gen(self, tac):
        start, end = new_tmp(), new_tmp()
        tac.append(f"label {start}")
        c = self.cond.gen(tac)
        tac.append(f"ifFalse {c} goto {end}")
        self.body.gen(tac)
        tac.append(f"goto {start}")
        tac.append(f"label {end}")

# -----------------------------
# PARSER + SEMANTIC ANALYSIS + RECOVERY
# -----------------------------
class MiniParser:
    """
    Parser for the mini-language. Builds an AST and performs basic semantic checks.
    Implements error recovery for missing semicolons and other syntax errors.
    """
    tokens = MiniLexer.tokens
    precedence = (
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('right', 'UMINUS'),
    )

    def __init__(self, lexer: MiniLexer):
        self.lexer = lexer
        self.symtab_stack: List[Dict[str, str]] = [{}]  # Symbol table stack
        self.errors: List[str] = []
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)

    def p_program(self, p):
        "program : stmt_list"
        p[0] = Seq(p[1])

    def p_stmt_list(self, p):
        """stmt_list : stmt_list stmt
                     | empty"""
        p[0] = p[1] + [p[2]] if len(p) == 3 else []

    def p_stmt(self, p):
        """stmt : assign_stmt
                | if_stmt
                | while_stmt"""
        p[0] = p[1]

    def p_assign(self, p):
        "assign_stmt : ID ASSIGN expr SEMI"
        self.symtab_stack[-1][p[1]] = 'int'
        p[0] = Assign(p[1], p[3])

    def p_if(self, p):
        "if_stmt : IF LPAREN expr RPAREN LBRACE stmt_list RBRACE"
        p[0] = If(p[3], Seq(p[6]))

    def p_while(self, p):
        "while_stmt : WHILE LPAREN expr RPAREN LBRACE stmt_list RBRACE"
        p[0] = While(p[3], Seq(p[6]))

    def p_expr_bin(self, p):
        """expr : expr PLUS expr
                | expr MINUS expr
                | expr TIMES expr
                | expr DIVIDE expr"""
        p[0] = BinOp(p[2], p[1], p[3])

    def p_expr_number(self, p):
        "expr : NUMBER"
        p[0] = Number(p[1])

    def p_expr_id(self, p):
        "expr : ID"
        if not self.lookup(p[1]):
            self.errors.append(
                f"Semantic warning: undeclared variable '{p[1]}' implicitly declared int (line {p.lineno(1)})")
            self.symtab_stack[-1][p[1]] = 'int'
        p[0] = Identifier(p[1])

    def p_expr_group(self, p):
        "expr : LPAREN expr RPAREN"
        p[0] = p[2]

    def p_expr_uminus(self, p):
        "expr : MINUS expr %prec UMINUS"
        p[0] = BinOp('-', Number(0), p[2])

    def p_empty(self, p):
        'empty :'
        p[0] = []

    def p_error(self, tok):
        """
        Error recovery for the parser. Handles missing semicolons and skips to sync tokens.
        """
        if not tok:
            self.errors.append("Syntax error: unexpected EOF")
            return

        # Heuristic: treat 'missing ;' before a new statement
        stmt_starters = ('ID', 'IF', 'WHILE', 'RBRACE')
        if tok.type in stmt_starters:
            self.errors.append(
                f"Inserted missing ';' before '{tok.value}' (line {tok.lineno})")
            self.parser.errok()  # continue with current token
            return

        # Generic panic-mode: discard until a sync token
        self.errors.append(
            f"Syntax error at '{tok.value}' (line {tok.lineno}), skipping tokens…")
        sync = ('SEMI', 'RBRACE')
        while True:
            tok = self.parser.token()
            if not tok or tok.type in sync:
                break
        self.parser.errok()

    def lookup(self, name: str) -> bool:
        """Check if a variable is declared in any scope."""
        return any(name in scope for scope in reversed(self.symtab_stack))

    def parse(self, text: str):
        """
        Parse the input text and return the AST and any errors found.
        """
        self.errors.clear()
        self.symtab_stack = [{}]
        ast = self.parser.parse(text, lexer=self.lexer.lexer, debug=False)
        return ast, self.errors.copy()

# -----------------------------
# INTERPRETER (TAC executor)
# -----------------------------
class Interpreter:
    """
    Simple interpreter for executing Three Address Code (TAC).
    Maintains an environment mapping variable names to integer values.
    """
    def __init__(self):
        self.env: Dict[str, int] = {}

    def run(self, tac: List[str]):
        pc = 0
        labels = {}
        # First pass: collect label positions
        for i, inst in enumerate(tac):
            if inst.startswith('label'):
                labels[inst.split()[1]] = i
        # Second pass: execute instructions
        while pc < len(tac):
            inst = tac[pc]
            if inst.startswith('label'):
                pc += 1
                continue
            if inst.startswith('ifFalse'):
                _, cond, _, lbl = inst.split()
                if not self._val(cond):
                    pc = labels[lbl]
                    continue
            elif inst.startswith('goto'):
                pc = labels[inst.split()[1]]
                continue
            else:
                left, expr = (s.strip() for s in inst.split('=', 1))
                if any(op in expr for op in '+-*/'):
                    b, op, c = expr.split()
                    self.env[left] = self._apply(op, self._val(b), self._val(c))
                else:
                    self.env[left] = self._val(expr)
            pc += 1
        return self.env

    def _val(self, token: str) -> int:
        if token.lstrip('-').isdigit():
            return int(token)
        return self.env.get(token, 0)

    def _apply(self, op, a, b):
        return {'+': a + b, '-': a - b, '*': a * b, '/': a // b}[op]

# -----------------------------
# GUI
# -----------------------------
class MiniGUI(tk.Tk):
    """
    Graphical User Interface for the mini-language compiler.
    Allows users to enter code, compile, and view tokens, errors, TAC, and output.
    """
    def __init__(self):
        super().__init__()
        self.title("Mini Compiler")
        self.geometry("1200x700")
        self._build()
        self.lexer = MiniLexer()
        self.parser = MiniParser(self.lexer)

    def _build(self):
        # Layout: left = code input, right = output
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.code = tk.Text(left, wrap='none')
        self.code.pack(fill=tk.BOTH, expand=True)
        self.code.insert('1.0', "// Sample\n\na=2+3*4\nwhile(a-10){\n  a=a-1\n}\n")
        ttk.Button(left, text="Compile & Run", command=self.compile_run).pack(fill=tk.X)

        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Tokens").pack()
        self.tokens = ttk.Treeview(right, columns=('type', 'val', 'line'),
                                   show='headings', height=15)
        for c in ('type', 'val', 'line'):
            self.tokens.heading(c, text=c)
        self.tokens.pack()
        ttk.Label(right, text="Errors / TAC / Output").pack()
        self.log = tk.Text(right, height=20, fg='blue')
        self.log.pack(fill=tk.BOTH, expand=True)

    def compile_run(self):
        """
        Compile and run the code entered in the text box.
        Shows tokens, errors, TAC, and final environment.
        """
        src = self.code.get('1.0', tk.END)
        toks, lerr = self.lexer.scan(src)

        self.tokens.delete(*self.tokens.get_children())
        for t in toks:
            self.tokens.insert('', tk.END, values=(t[0], t[1], t[2]))

        ast, perr = self.parser.parse(src)
        self.log.delete('1.0', tk.END)

        if lerr or perr:
            for e in lerr + perr:
                self.log.insert(tk.END, e + '\n')
            if not ast:
                return  # unrecoverable
        tac = []
        ast.gen(tac)
        self.log.insert(tk.END, 'TAC:\n' + '\n'.join(tac) + '\n\n')
        env = Interpreter().run(tac)
        self.log.insert(tk.END, 'Final env: ' + str(env))

# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    if '--test' in sys.argv:
        import unittest

        class CompilerTest(unittest.TestCase):
            """Unit tests for the mini-language compiler/interpreter."""
            def _compile(self, src):
                lexr, parser = MiniLexer(), MiniParser(MiniLexer())
                ast, errs = parser.parse(src)
                self.assertFalse(errs)
                tac = []
                ast.gen(tac)
                return Interpreter().run(tac)

            def test_arith(self):
                e = self._compile("a=2+3*4; b=a-5;")
                self.assertEqual((e['a'], e['b']), (14, 9))

            def test_if(self):
                e = self._compile("x=1; if(x){y=10;}")
                self.assertEqual(e['y'], 10)

            def test_while(self):
                e = self._compile("i=0; while(i-5){ i=i+1; }")
                self.assertEqual(e['i'], 5)

        unittest.main(argv=[sys.argv[0]])
    elif len(sys.argv) == 2 and sys.argv[1] not in ('--test',):
        # Run as a command-line compiler
        source = Path(sys.argv[1]).read_text(encoding='utf-8')
        lexr, parser = MiniLexer(), MiniParser(MiniLexer())
        ast, errs = parser.parse(source)
        if errs:
            print('\n'.join(errs))
            sys.exit(1)
        tac = []
        ast.gen(tac)
        print("TAC:\n" + '\n'.join(tac))
        print("\nFinal environment:", Interpreter().run(tac))
    else:
        # Launch the GUI
        MiniGUI().mainloop()