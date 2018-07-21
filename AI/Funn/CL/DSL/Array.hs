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
module AI.Funn.CL.DSL.Array (Index(..), (!), at) where

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
import           AI.Funn.CL.DSL.Code
import           AI.Funn.Space

class Index x i a | x -> i, x -> a where
  at :: x -> i -> Expr a

instance Index (Array m a) (Expr Int) a where
  at (Array name) (Expr i) = Expr (AST.ExprIndex (name ++ "_base") index)
    where
      offset = AST.ExprVar (name ++ "_offset")
      index = AST.ExprOp offset "+" i

infixl 9 !
(!) :: Index x i a => x -> i -> Expr a
(!) = at
