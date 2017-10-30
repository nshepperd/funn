{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeOperators #-}
module AI.Funn.CL.DSL.Code where

import           Control.Monad
import           Control.Monad.Free
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Ratio
import           Data.Traversable
import           Foreign.Ptr
import           Foreign.Storable

import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.Space

newtype Var a = Var { varName :: AST.Name }
newtype Expr a = Expr AST.Expr

class Variable a where
  declare :: Var a -> AST.Decl

instance Variable Int where
  declare (Var name) = AST.Decl "int" name

instance Variable Float where
  declare (Var name) = AST.Decl "float" name

instance Variable Double where
  declare (Var name) = AST.Decl "double" name

  -- OpenCL Functions
  -- Argument types

exprVar :: Var a -> Expr a
exprVar (Var name) = Expr (AST.ExprVar name)

class Argument a where
  declareArgument :: AST.Name -> (a, [AST.Decl])

instance Variable a => Argument (Expr a) where
  declareArgument argName = (exprVar var, [declare var])
    where
      var = Var (argName)

  -- Function -> Kernel

newtype KernelSource (xs :: [*]) = KernelSource String
  deriving Show

class ToKernel a xs | a -> xs, xs -> a where
  toKernel :: Int -> a -> AST.Kernel
  kernel :: a -> KernelSource xs

instance ToKernel (CL ()) '[] where
  toKernel n cl = AST.Kernel "run" [] (runCL cl)
  kernel cl = KernelSource (AST.printKernel (toKernel 0 cl))

instance (Argument a, ToKernel r rs) => ToKernel (a -> r) (a : rs) where
  toKernel n f = let (a, args) = declareArgument ("arg" ++ show n)
                     AST.Kernel name args' body = toKernel (n+1) (f a)
                 in AST.Kernel name (args ++ args') body
  kernel cl = KernelSource (AST.printKernel (toKernel 0 cl))

kernelStr :: ToKernel a xs => a -> String
kernelStr a = AST.printKernel (toKernel 0 a)

  -- Operational Monad

type CL = Free CLF
data CLF (r :: *) where
  NewVar :: Variable a => (Expr a -> r) -> CLF r
  NewVarAssign :: Variable a => Expr a -> (Expr a -> r) -> CLF r
  Assign :: Expr a -> Expr a -> r -> CLF r
  ForLoop :: Expr Int -> Expr Int -> (Expr Int -> CL ()) -> r -> CLF r
deriving instance Functor CLF

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

runCL :: CL () -> AST.Block
runCL cl = AST.Block (runGenSym (go cl))
  where
    go (Pure ()) = return []
    go (Free clf) = case clf of
      NewVar k -> do
        var <- genvar
        rest <- go (k (exprVar var))
        return $ [AST.StmtDeclare (declare var)] <> rest
      NewVarAssign (Expr e) k -> do
        var <- genvar
        rest <- go (k (exprVar var))
        return $ [AST.StmtDeclareAssign (declare var) e] <> rest
      Assign (Expr v) (Expr e) k -> do
        rest <- go k
        return $ [AST.StmtAssign v e] <> rest
      ForLoop (Expr from) (Expr to) sub r -> do
        var <- genvar
        loop <- AST.Block <$> go (sub (exprVar var))
        rest <- go r
        return $ [AST.StmtDeclare (declare var), AST.StmtForEach (varName var) from to loop] <> rest

  -- User utilities

newvar :: Variable a => CL (Expr a)
newvar = liftF (NewVar id)

initvar :: Variable a => Expr a -> CL (Expr a)
initvar e = liftF (NewVarAssign e id)

forEach :: Expr Int -> Expr Int -> (Expr Int -> CL ()) -> CL ()
forEach from to sub = liftF (ForLoop from to sub ())

eval :: Variable a => Expr a -> CL (Expr a)
eval = initvar

infixl 1 .=
(.=) :: Expr a -> Expr a -> CL ()
v .= e = liftF (Assign v e ())

-- Embedding literals

operator :: String -> Expr a -> Expr a -> Expr a
operator op (Expr a) (Expr b) = Expr (AST.ExprOp a op b)

class Literal a where
  literal :: a -> Expr a

instance Literal Int where
  literal n = Expr $ AST.ExprLit (show n)

instance Num (Expr Int) where
  fromInteger n = literal (fromInteger n)
  abs = function "abs"
  signum = function "sign"
  negate n = 0 - n
  (+) = operator "+"
  (-) = operator "-"
  (*) = operator "*"

class Bitwise a where
  (.&.) :: a -> a -> a
  (.|.) :: a -> a -> a
  xor :: a -> a -> a

instance Bitwise (Expr Int) where
  (.&.) = operator "&"
  (.|.) = operator "|"
  xor = operator "^"

shiftL :: Expr Int -> Expr Int -> Expr Int
shiftL = operator "<<"

shiftR :: Expr Int -> Expr Int -> Expr Int
shiftR = operator ">>"

instance Literal Float where
  literal n = Expr $ AST.ExprLit (show n ++ "f")

instance Literal Double where
  literal n = Expr $ AST.ExprLit (show n)

instance Num (Expr Float) where
  fromInteger n = literal (fromInteger n)
  abs = function "fabs"
  signum = function "sign"
  negate n = 0 - n
  (+) = operator "+"
  (-) = operator "-"
  (*) = operator "*"

instance Num (Expr Double) where
  fromInteger n = literal (fromInteger n)
  abs = function "fabs"
  signum = function "sign"
  negate n = 0 - n
  (+) = operator "+"
  (-) = operator "-"
  (*) = operator "*"

instance Fractional (Expr Float) where
  (/) = operator "/"
  recip a = 1 / a
  fromRational x = fromInteger (numerator x) / fromInteger (denominator x)

instance Fractional (Expr Double) where
  (/) = operator "/"
  recip a = 1 / a
  fromRational x = fromInteger (numerator x) / fromInteger (denominator x)

type CLNum a = (Storable a, Num (Expr a), Literal a, Variable a, Argument (Array W a), Argument (Array R a))
type CLFractional a = (Storable a, Fractional (Expr a), Literal a,
                       Variable a, Argument (Array W a), Argument (Array R a))
type CLFloating a = (Storable a, Floating (Expr a), Literal a,
                      Variable a, Argument (Array W a), Argument (Array R a))
type CLFloats a = (Storable a, Floating (Expr a), Literal a,
                   Variable a, Argument (Array W a), Argument (Array R a),
                   Floats a)

constant :: String -> Expr a
constant name = Expr (AST.ExprLit name)

instance Floating (Expr Float) where
  pi = constant "M_PI_F"
  exp = function "exp"
  log = function "log"
  sqrt = function "sqrt"
  (**) = function "pow"
  logBase b e = log e / log b
  sin = function "sin"
  cos = function "cos"
  tan = function "tan"
  asin = function "asin"
  acos = function "acos"
  atan = function "atan"
  sinh = function "sinh"
  cosh = function "cosh"
  tanh = function "tanh"
  asinh = function "asinh"
  acosh = function "acosh"
  atanh = function "atanh"

instance Floating (Expr Double) where
  pi = constant "M_PI"
  exp = function "exp"
  log = function "log"
  sqrt = function "sqrt"
  (**) = function "pow"
  logBase b e = log e / log b
  sin = function "sin"
  cos = function "cos"
  tan = function "tan"
  asin = function "asin"
  acos = function "acos"
  atan = function "atan"
  sinh = function "sinh"
  cosh = function "cosh"
  tanh = function "tanh"
  asinh = function "asinh"
  acosh = function "acosh"
  atanh = function "atanh"

-- Builtin functions

class DefineFunction r where
  function_ :: AST.Name -> [AST.Expr] -> r

instance DefineFunction (Expr a) where
  function_ name args = Expr (AST.ExprCall name (reverse args))

instance DefineFunction r => DefineFunction (Expr a -> r) where
  function_ name args (Expr a) = function_ name (a : args)

function :: DefineFunction r => AST.Name -> r
function name = function_ name []

get_global_id :: Expr Int -> CL (Expr Int)
get_global_id i = eval (function "get_global_id" i)

class Relational a where
  fmin :: Expr a -> Expr a -> Expr a
  fmax :: Expr a -> Expr a -> Expr a
  fstep :: Expr a -> Expr a -> Expr a

instance Relational Float where
  fmin = function "min"
  fmax = function "max"
  fstep = function "step"

instance Relational Double where
  fmin = function "min"
  fmax = function "max"
  fstep = function "step"

data Mode = R | W
newtype Array (m :: Mode) a = Array AST.Name
type ArrayR = Array R
type ArrayW = Array W

instance Argument (Array R Float) where
  declareArgument argName = (Array argName, [base, offset])
    where
      base = AST.Decl "__global const float*" (argName ++ "_base")
      offset = AST.Decl "const int" (argName ++ "_offset")

instance Argument (Array W Float) where
  declareArgument argName = (Array argName, [base, offset])
    where
      base = AST.Decl "__global float*" (argName ++ "_base")
      offset = AST.Decl "const int" (argName ++ "_offset")

instance Argument (Array R Double) where
  declareArgument argName = (Array argName, [base, offset])
    where
      base = AST.Decl "__global const double*" (argName ++ "_base")
      offset = AST.Decl "const int" (argName ++ "_offset")

instance Argument (Array W Double) where
  declareArgument argName = (Array argName, [base, offset])
    where
      base = AST.Decl "__global double*" (argName ++ "_base")
      offset = AST.Decl "const int" (argName ++ "_offset")

at :: Array m a -> Expr Int -> Expr a
at (Array name) (Expr i) = Expr (AST.ExprIndex (name ++ "_base") index)
  where
    offset = AST.ExprVar (name ++ "_offset")
    index = AST.ExprOp offset "+" i
