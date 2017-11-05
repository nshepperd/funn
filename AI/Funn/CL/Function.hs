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
module AI.Funn.CL.Function (clfun, CLType(..), KernelProgram(..), compile, KTable, newKTable, memoc, memo) where

import           Control.Monad
import           Control.Monad.Free
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.IORef
import           Data.Int
import           Data.List
import           Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import           Data.Monoid
import           Data.Proxy
import           Data.Ratio
import           Data.Traversable
import           Foreign.Ptr
import           Foreign.Storable
import           GHC.Stack
import           System.IO.Unsafe

import           AI.Funn.CL.DSL.Code
import           AI.Funn.CL.MonadCL
import qualified Foreign.OpenCL.Bindings as CL

newtype KernelProgram (xs :: [*]) = KernelProgram CL.Kernel

compileIO :: MonadIO m => KernelSource xs -> m (KernelProgram xs)
compileIO (KernelSource src) = liftIO $ do
  putStrLn $ "Compiling program: " ++ src
  ctx <- getContext
  deviceID <- getDeviceID
  prog <- CL.createProgram ctx src
  CL.buildProgram prog [deviceID] ""
  k <- CL.createKernel prog "run"
  return (KernelProgram k)

{-# NOINLINE compile #-}
compile :: ToKernel f xs => f -> KernelProgram xs
compile f = unsafePerformIO (compileIO (kernel f))

newtype KTable k = KTable (IORef (Map k CL.Kernel))

{-# NOINLINE newKTable #-}
newKTable :: Ord k => (IO (KTable k) -> KTable k) -> KTable k
newKTable f = f (KTable <$> newIORef Map.empty)

memoc :: (HasCallStack, ToKernel f xs, RunKernel xs g, Ord k)
      => KTable k -> k -> f -> [Int] -> g
memoc table key f = clfun (memo table key (compile f))

{-# NOINLINE memo #-}
memo :: Ord k => KTable k -> k -> KernelProgram xs -> KernelProgram xs
memo (KTable memoTable) key (KernelProgram prog) = unsafePerformIO $ do
  oldMap <- readIORef memoTable
  case Map.lookup key oldMap of
    Just p -> return (KernelProgram p)
    Nothing -> do
      writeIORef memoTable (Map.insert key prog oldMap)
      return (KernelProgram prog)


class Argument x => CLType a x | a -> x where
  karg :: HasCallStack => a -> KernelArg

instance CLType Float (Expr Float) where
  karg x = KernelArg (\f -> f [CL.VArg x])
instance CLType Double (Expr Double) where
  karg x = KernelArg (\f -> f [CL.VArg x])
instance CLType Int (Expr Int) where
  karg x = KernelArg (\f -> f [CL.VArg (fromIntegral x :: Int32)])

-- Unsafe. Exists only so RunKernel can be recursive.
sub :: KernelProgram (x:xs) -> KernelProgram xs
sub (KernelProgram k) = KernelProgram k

class RunKernel xs f | f -> xs where
  func :: HasCallStack => KernelProgram xs -> ([Int] -> (KernelArg -> KernelArg) -> f)

instance RunKernel '[] (IO ()) where
  {-# INLINE func #-}
  func (KernelProgram k) ranges prependArgs = runCompiled k [prependArgs mempty] [] (map fromIntegral ranges) []

instance (RunKernel xs f, CLType a x) => RunKernel (x:xs) (a -> f) where
  {-# INLINE func #-}
  func kp ranges prependArgs = \a -> func (sub kp) ranges (prependArgs . (karg a<>))

{-# INLINE clfun #-}
clfun :: (HasCallStack, RunKernel xs f) => KernelProgram xs -> [Int] -> f
clfun g ranges = func g ranges id
