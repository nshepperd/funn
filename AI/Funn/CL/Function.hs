{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.CL.Function (clfun, CLType(..)) where

import Control.Monad
import Control.Monad.Free
import Data.Foldable
import Data.Int
import Data.List
import Data.Monoid
import Data.Traversable
import Data.Proxy
import Data.Ratio
import Foreign.Ptr
import Foreign.Storable

import AI.Funn.CL.Code
import AI.Funn.CL.MonadCL
import qualified Foreign.OpenCL.Bindings as CL

class Argument x => CLType a x | a -> x where
  karg :: a -> KernelArg

instance CLType Float (Expr Float) where
  karg x = KernelArg (\f -> f [CL.VArg x])
instance CLType Double (Expr Double) where
  karg x = KernelArg (\f -> f [CL.VArg x])
instance CLType Int (Expr Int) where
  karg x = KernelArg (\f -> f [CL.VArg (fromIntegral x :: Int32)])

data KFun f where
  Arg :: (a -> KFun as) -> KFun (a -> as)
  Done :: KernelArg -> (KernelArg -> [Int] -> IO ()) -> KFun (IO ())

consArg :: KernelArg -> KFun xs -> KFun xs
consArg arg (Done as run) = Done (arg <> as) run
consArg arg (Arg f) = Arg (consArg arg . f)

class ToKernel g => CLFunc as g | as -> g, as -> where
  func :: String -> KFun as
  func' :: g -> KFun as

instance CLFunc (IO ()) (CL ()) where
  func src = Done mempty (\args ranges -> runKernel src "run" [args] [] (map fromIntegral ranges) [])
  func' g = func (kernel g)

instance (CLFunc as gr, CLType a x) => CLFunc (a -> as) (x -> gr) where
  func src = Arg (\a -> consArg (karg a) next)
    where
      next = func src
  func' g = func (kernel g)

runfun :: KFun f -> [Int] -> f
runfun (Done args run) range = run args range
runfun (Arg k) range = \a -> runfun (k a) range

clfun :: CLFunc f g => g -> [Int] -> f
clfun g = runfun k
  where
    k = func' g
