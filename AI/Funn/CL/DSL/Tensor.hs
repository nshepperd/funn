{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
module AI.Funn.CL.DSL.Tensor (TensorCL, MTensorCL, CTensor(..), dimsOf, splitIndex, indexFlat) where

import           Control.Monad
import           Control.Monad.Free
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits

import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code
import           AI.Funn.Space

data CTensor (m :: Mode) (ds :: [Nat]) = CTensor (DimDict ds) (Array m Double)
  deriving Show

type TensorCL = CTensor R
type MTensorCL = CTensor W

data DimDict (ds :: [Nat]) where
  DimZero :: DimDict '[]
  DimSucc :: Expr Int -> DimDict ds -> DimDict (d ': ds)

instance Show (DimDict ds) where
  showsPrec n DimZero = showString "DimZero"
  showsPrec n (DimSucc d ds) = showParen (n > 10) $
    showString "DimSucc " . showsPrec 11 d . showString " " . showsPrec 11 ds

declareDict :: AST.Name -> Int -> KnownListOf ds -> (DimDict ds, [AST.Decl])
declareDict name n KnownListNil = (DimZero, [])
declareDict name n (KnownListCons xs) = (DimSucc d ds, darg ++ dsargs)
  where
      (ds, dsargs) = declareDict name (n+1) xs
      (d, darg) = declareArgument (name ++ "_" ++ show n)


instance KnownList ds => Argument (DimDict ds) where
  declareArgument name = declareDict name 0 (knownList (Proxy @ ds))

instance (Argument (Array m Double), KnownList ds) => Argument (CTensor m ds) where
  declareArgument name = (CTensor dim arr, dim_args ++ arr_args)
    where
      (dim, dim_args) = declareArgument name
      (arr, arr_args) = declareArgument name

instance Index (CTensor m ds) [Expr Int] Double where
  at = index

dimsOf :: CTensor m ds -> [Expr Int]
dimsOf (CTensor ds _) = go ds
  where
    go :: DimDict xs -> [Expr Int]
    go DimZero = []
    go (DimSucc d ds) = d : go ds

index :: CTensor m ds -> [Expr Int] -> Expr Double
index (CTensor (DimSucc _ ds) arr) (i:is) = go ds arr is i
  where
    go :: DimDict xs -> Array m Double -> [Expr Int] -> Expr Int -> Expr Double
    go DimZero arr [] ix = at arr ix
    go (DimSucc d ds) arr (i:is) ix = go ds arr is (ix * d + i)

indexFlat :: CTensor m ds -> Expr Int -> Expr Double
indexFlat (CTensor _ arr) i = at arr i

splitIndex :: [Expr Int] -> Expr Int -> CL [Expr Int]
splitIndex ds i = reverse <$> go (reverse $ tail ds) i
  where
    go (d:ds) i = do a <- eval (i `mod'` d)
                     i' <- eval (i `div'` d)
                     (a:) <$> go ds i'
    go [] i = return [i]
