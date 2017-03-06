{-# LANGUAGE GADTs #-}

module AI.Funn.CL.AST where

import Data.List.Split
import Data.List

type Name = String
type TypeDecl = String

data Expr = ExprLit String
          | ExprVar Name
          | ExprIndex Name Expr
          | ExprCall Name [Expr]
          | ExprOp Expr Name Expr
  deriving (Show)

data Decl = Decl TypeDecl Name
  deriving (Show)

data Stmt = StmtAssign Expr Expr
          | StmtDeclare Decl
          | StmtDeclareAssign Decl Expr
          | StmtForEach Name Expr Expr Block
  deriving (Show)

data Block = Block [Stmt]
  deriving (Show)

data Kernel = Kernel Name [Decl] Block
  deriving (Show)

indent :: Int -> String -> String
indent n str = intercalate "\n" (fmap go ls)
  where
    ls = splitOn "\n" str
    go line
      | null line = line
      | otherwise = replicate n ' ' ++ line

printExpr :: Expr -> String
printExpr (ExprLit s) = s
printExpr (ExprVar v) = v
printExpr (ExprIndex v i) = v ++ "[" ++ printExpr i ++ "]"
printExpr (ExprCall f args) = f ++ "(" ++ intercalate ", " (map printExpr args) ++ ")"
printExpr (ExprOp a x b) = "(" ++ printExpr a ++ " " ++ x ++ " " ++ printExpr b ++ ")"

printDecl :: Decl -> String
printDecl (Decl t n) = t ++ " " ++ n

printStmt :: Stmt -> String
printStmt (StmtAssign v e) = printExpr v ++ " = " ++ printExpr e ++ ";"
printStmt (StmtDeclare d) = printDecl d ++ ";"
printStmt (StmtDeclareAssign d e) = printDecl d ++ " = " ++ printExpr e ++ ";"
printStmt (StmtForEach name start end body) = "for ("
                                               ++ name ++ " = " ++ printExpr start ++ "; "
                                               ++ name ++ " < " ++ printExpr end ++ "; "
                                               ++ name ++ "++) {\n"
                                               ++ indent 2 (printBlock body)
                                               ++ "}"

printBlock :: Block -> String
printBlock (Block statements) = unlines (map printStmt statements)

printKernel :: Kernel -> String
printKernel (Kernel name args body) =
  "__kernel void " ++ name
  ++ "(" ++ intercalate ", " (map printDecl args) ++ ") {\n"
  ++ indent 2 (printBlock body)
  ++ "}"
