{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.CL.Code where

import Control.Monad
import Control.Monad.Free
import Data.Foldable
import Data.List
import Data.Monoid
import Data.Traversable
import Data.Ratio
import Foreign.Ptr

  -- AST

newtype Var a = Var { varName :: String } deriving (Show, Eq)

data Op a = Op String deriving (Show, Eq)

data Expr a where
  ExpLit :: String -> Expr a
  ExpIndex :: Expr (Ptr a) -> Expr Int -> Expr a
  ExpCall :: Var (a -> b) -> Expr a -> Expr b
  ExpBinOp :: Op (a -> b -> c) -> Expr a -> Expr b -> Expr c
  ExpVar :: Var a -> Expr a

data Statement where
  StmtReturn :: Statement
  StmtAssign :: Expr a -> Expr a -> Statement
  StmtNewVar :: Variable a => Var a -> Statement
  StmtFor :: Var Int -> Expr Int -> Expr Int -> Block -> Statement

data Block where
  Block :: [Statement] -> Block

instance Monoid Block where
  mempty = Block []
  mappend (Block xs) (Block ys) = Block (xs <> ys)

data Kernel where
  Kernel :: String -> [String] -> Block -> Kernel

deriving instance Show (Expr a)
deriving instance Show Statement
deriving instance Show Block
deriving instance Show Kernel

  -- Printing as OpenCL code.

printKernel :: Kernel -> String
printKernel (Kernel name args body) = "__kernel void " ++ name ++ "(" ++ showArgs args ++ ") {\n"
                                      ++ indent 2 (printBlock body)
                                      ++ "}\n"
  where
    showArgs args = intercalate ", " args

indent :: Int -> String -> String
indent n str = unlines (fmap (replicate n ' ' ++) (lines str))

printBlock :: Block -> String
printBlock (Block ss) = unlines $ fmap printStatement ss

printStatement :: Statement -> String
printStatement StmtReturn = "return;"
printStatement (StmtAssign expl expr) = printExpr expl ++ " = " ++ printExpr expr ++ ";"
printStatement (StmtNewVar var) = declare var ++ ";"
printStatement (StmtFor var from to each) =
  "for (" ++ ("int " ++ varName var ++ " = " ++ printExpr from ++ "; " ++
               varName var ++ " < " ++ printExpr to ++ "; " ++
               varName var ++ "++") ++ ") {\n" ++
  indent 2 (printBlock each)
  ++ "}"

printExpr :: Expr a -> String
printExpr (ExpVar (Var name)) = name
printExpr (ExpLit lit) = lit
printExpr (ExpIndex arr i) = printExpr arr ++ "[" ++ printExpr i ++ "]"
printExpr (ExpCall (Var fun) expr) = fun ++ "(" ++ printExpr expr ++ ")"
printExpr (ExpBinOp (Op op) x y) = paren (printExpr x ++ op ++ printExpr y)
  where
    paren xs = "(" ++ xs ++ ")"

  -- Definable variables

class Variable a where
  declare :: Var a -> String

instance Variable Int where
  declare (Var name) = "int " ++ name

instance Variable Float where
  declare (Var name) = "float " ++ name

  -- OpenCL Functions
  -- Argument types

class Argument a where
  declareArgument :: String -> (a, [String])

instance Argument (Var Int) where
  declareArgument argName = (Var argName, ["int " ++ argName])
instance Argument (Expr Int) where
  declareArgument argName = (ExpVar (Var argName), ["const int " ++ argName])

instance Argument (Expr Float) where
  declareArgument argName = (ExpVar (Var argName), ["const float " ++ argName])

  -- Function -> Kernel

class ToKernel a where
  toKernel :: Int -> a -> Kernel

instance ToKernel (CL ()) where
  toKernel n cl = Kernel "run" [] (runCL cl)

instance (Argument a, ToKernel r) => ToKernel (a -> r) where
  toKernel n f = let (var, args) = declareArgument ("arg" ++ show n)
                     Kernel name args' body = toKernel (n+1) (f var)
                 in Kernel name (args ++ args') body

kernel :: ToKernel a => a -> String
kernel a = printKernel (toKernel 0 a)

  -- Operational Monad

type CL = Free CLF
data CLF (r :: *) where
  NewVar :: Variable a => (Var a -> r) -> CLF r
  Assign :: Expr a -> Expr a -> r -> CLF r
  Return :: CLF r
  ForLoop :: Expr Int -> Expr Int -> (Var Int -> CL ()) -> r -> CLF r
deriving instance Functor CLF

newvar :: Variable a => CL (Var a)
newvar = liftF (NewVar id)

infixl 1 .=
(.=) :: Var a  -> Expr a -> CL ()
l .= r = liftF (Assign (ExpVar l) r ())

ret :: CL ()
ret = liftF Return

  -- Embedding literals

class Literal a where
  literal :: a -> Expr a

instance Literal Int where
  literal n = ExpLit (show n)

instance Literal Float where
  literal n = ExpLit (show n)

instance (Literal a, Num a) => Num (Expr a) where
  fromInteger n = literal (fromInteger n)
  abs = undefined
  signum = undefined
  negate n = 0 - n
  (+) a b = ExpBinOp (Op "+") a b
  (-) a b = ExpBinOp (Op "-") a b
  (*) a b = ExpBinOp (Op "*") a b

instance (Literal a, Fractional a) => Fractional (Expr a) where
  a / b = ExpBinOp (Op "/") a b
  recip a = 1 / a
  fromRational x = fromInteger (numerator x) / fromInteger (denominator x)

  -- Gensym monad for supply of fresh variable names

data GenSymF r where
  GenSym :: (String -> r) -> GenSymF r
deriving instance (Functor GenSymF)
type GenSym a = Free GenSymF a

gensym :: GenSym String
gensym = liftF (GenSym id)
genvar :: GenSym (Var a)
genvar = Var <$> gensym

runGenSym :: GenSym a -> a
runGenSym m = go 0 m
  where
    go n (Pure a) = a
    go n (Free (GenSym k)) = go (n+1) (k ("var" ++ show n))

  -- Compile to AST

runCL :: CL () -> Block
runCL cl = runGenSym (go cl)
  where
    go :: CL () -> GenSym Block
    go (Pure ()) = return (Block [])
    go (Free clf) = case clf of
      NewVar k -> do
        var <- genvar
        rest <- go (k var)
        return $ Block [StmtNewVar var] <> rest
      Assign l r k -> do
        rest <- go k
        return $ Block [StmtAssign l r] <> rest
      Return -> return (Block [StmtReturn])
      ForLoop from to sub r -> do
        var <- genvar
        loop <- go (sub var)
        rest <- go r
        return $ Block [StmtFor var from to loop] <> rest

  -- User utilities

get_global_id :: Expr Int -> CL (Expr Int)
get_global_id dim = do v <- initVar (ExpCall (Var "get_global_id") dim)
                       return (get v)

sqrtf :: Expr Float -> Expr Float
sqrtf x = ExpCall (Var "sqrt") x

forLoop :: Expr Int -> Expr Int -> (Expr Int -> CL ()) -> CL ()
forLoop from to sub = Free (ForLoop from to (sub . get) (Pure ()))

get :: Var a -> Expr a
get = ExpVar

readVar :: Variable a => Var a -> CL (Expr a)
readVar var = do w <- newvar
                 w .= get var
                 return (get w)

initVar :: Variable a => Expr a -> CL (Var a)
initVar expr = do v <- newvar
                  v .= expr
                  return v

readPtr :: Var (Ptr a) -> Expr Int -> Expr a
readPtr var i = ExpIndex (ExpVar var) i

writePtr :: Var (Ptr a) -> Expr Int -> Expr a -> CL ()
writePtr var i a = liftF (Assign (ExpIndex (ExpVar var) i) a ())

data Mode = R | W
data Arr (m :: Mode) a = Arr (Var (Ptr a)) (Expr Int)

instance Argument (Arr R Float) where
  declareArgument argName = (Arr (Var argName)
                              (ExpVar (Var (argName ++ "_offset"))),
                             ["__global const float* " ++ argName,
                              "const int " ++ argName ++ "_offset"])

instance Argument (Arr W Float) where
  declareArgument argName = (Arr (Var argName)
                              (ExpVar (Var (argName ++ "_offset"))),
                             ["__global float* " ++ argName,
                              "const int " ++ argName ++ "_offset"])

readArr :: Arr m a -> Expr Int -> Expr a
readArr (Arr ptr offset) i = readPtr ptr (offset + i)
writeArr :: Arr W a -> Expr Int -> Expr a -> CL ()
writeArr (Arr ptr offset) i a = writePtr ptr (offset + i) a

-- arrSize :: Arr m a -> Expr Int
-- arrSize (Arr ptr offset size) = size

-- foo :: Arr W Float -> CL ()
-- foo arr = do x <- initVar 0
--              forLoop 0 (arrSize arr) $ \i -> do
--                writeArr arr i (get x)
--                x .= get x + 1
