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

class Argument x => CLType a x s | a -> x where
  karg :: a -> KernelArg s

instance CLType Float (Expr Float) s where
  karg x = KernelArg (\f -> f [CL.VArg x])
instance CLType Double (Expr Double) s where
  karg x = KernelArg (\f -> f [CL.VArg x])
instance CLType Int (Expr Int) s where
  karg x = KernelArg (\f -> f [CL.VArg (fromIntegral x :: Int32)])

data KFun s f where
  Arg :: (a -> KFun s as) -> KFun s (a -> as)
  Done :: KernelArg s -> (KernelArg s -> [Int] -> OpenCL s ()) -> KFun s (OpenCL s ())

consArg :: KernelArg s -> KFun s xs -> KFun s xs
consArg arg (Done as run) = Done (arg <> as) run
consArg arg (Arg f) = Arg (consArg arg . f)

class ToKernel g => CLFunc s as g | as -> g, as -> s where
  func :: String -> KFun s as
  func' :: g -> KFun s as

instance CLFunc s (OpenCL s ()) (CL ()) where
  func src = Done mempty (\args ranges -> runKernel src "run" [args] [] (map fromIntegral ranges) [])
  func' g = func (kernel g)

instance (CLFunc s as gr, CLType a x s) => CLFunc s (a -> as) (x -> gr) where
  func src = Arg (\a -> consArg (karg a) next)
    where
      next = func src
  func' g = func (kernel g)

runfun :: KFun s f -> [Int] -> f
runfun (Done args run) range = run args range
runfun (Arg k) range = \a -> runfun (k a) range

clfun :: CLFunc s f g => g -> [Int] -> f
clfun g = runfun k
  where
    k = func' g
