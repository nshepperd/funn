{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.CL.MonadCL (
  OpenCL, runOpenCL,
  getContext, getCommandQueue, getDevice,
  createKernel, KernelArg(..), runKernel,
  doubleArg, int32Arg) where

import           Control.Applicative
import           Control.Monad
import           Data.Foldable
import           Data.Traversable
import           Data.Monoid
import           Data.Int

import           Control.Concurrent.MVar
import           Data.Proxy
import           Data.Random

import           Control.Lens
import           Control.Monad.IO.Class
import           Control.Monad.Reader
import           Control.Monad.State.Lazy
import           Control.Monad.Trans

import           Data.Map (Map)
import qualified Data.Map.Strict as Map

import           Foreign.Ptr
import           GHC.Float
import           System.IO.Unsafe

import qualified Foreign.OpenCL.Bindings as CL
import qualified Foreign.OpenCL.Bindings.Synchronization as CL

data R = R {
  _platformID :: CL.PlatformID,
  _context :: CL.Context,
  _device :: CL.DeviceID,
  _queue :: CL.CommandQueue
  }
platformID :: Lens' R CL.PlatformID
platformID f (R a b c d) = (\a' -> R a' b c d) <$> f a
context :: Lens' R CL.Context
context f (R a b c d) = (\b' -> R a b' c d) <$> f b
device :: Lens' R CL.DeviceID
device f (R a b c d) = (\c' -> R a b c' d) <$> f c
queue :: Lens' R CL.CommandQueue
queue f (R a b c d) = (\d' -> R a b c d') <$> f d

data S = S {
  _programs :: Map String CL.Program,
  _kernels :: Map (String, String) CL.Kernel
  }
programs :: Lens' S (Map String CL.Program)
programs f (S x y) = (\x' -> S x' y) <$> f x
kernels :: Lens' S (Map (String,String) CL.Kernel)
kernels f (S x y) = S x <$> (f y)

newtype OpenCL s a = OpenCL { getOpenCL :: ReaderT R (StateT S IO) a }
                     deriving (Functor, Applicative, Monad, MonadIO)

runOpenCL :: (forall s. OpenCL s a) -> IO a
runOpenCL (OpenCL k) = do plat:_ <- CL.getPlatformIDs
                          [dev] <- CL.getDeviceIDs [CL.DeviceTypeAll] plat
                          ctx <- CL.createContext [dev] [CL.ContextPlatform plat] CL.NoContextCallback
                          q <- CL.createCommandQueue ctx dev []
                          evalStateT (runReaderT k (R plat ctx dev q)) (S Map.empty Map.empty)

getProgram :: String -> OpenCL s CL.Program
getProgram source = do ctx <- OpenCL $ view context
                       dev <- OpenCL $ view device
                       prg <- OpenCL $ use (programs . at source)
                       case prg of
                        Just program -> return program
                        Nothing -> do program <- liftIO $ CL.createProgram ctx source
                                      liftIO $ CL.buildProgram program [dev] ""
                                      OpenCL (programs . at source .= Just program)
                                      return program

createKernel :: String -> String -> OpenCL s CL.Kernel
createKernel source entryPoint = do program <- getProgram source
                                    liftIO $ CL.createKernel program entryPoint

getKernel :: String -> String -> OpenCL s CL.Kernel
getKernel source entryPoint = do k <- OpenCL $ use (kernels . at (source,entryPoint))
                                 case k of
                                  Just kernel -> return kernel
                                  Nothing -> do kernel <- createKernel source entryPoint
                                                OpenCL (kernels . at (source,entryPoint) .= Just kernel)
                                                return kernel

getContext :: OpenCL s CL.Context
getContext = OpenCL (view context)

getDevice :: OpenCL s CL.DeviceID
getDevice = OpenCL (view device)

getCommandQueue :: OpenCL s CL.CommandQueue
getCommandQueue = OpenCL (view queue)

newtype KernelArg s = KernelArg (([CL.KernelArg] -> IO ()) -> IO ())

instance Monoid (KernelArg s) where
  mempty = KernelArg (\k -> k [])
  mappend (KernelArg f1) (KernelArg f2) = KernelArg $ \k ->
    f1 (\args1 ->
          f2 (\args2 -> k (args1 ++ args2)))


doubleArg :: Double -> KernelArg s
doubleArg x = KernelArg run
  where
    run f = f [CL.VArg (double2Float x)]

int32Arg :: Integral a => a -> KernelArg s
int32Arg a = KernelArg run
  where
    run f = f [CL.VArg (fromIntegral a :: Int32)]

runKernel :: String -> String  -> [KernelArg s] -> [CL.ClSize] -> [CL.ClSize] -> [CL.ClSize] -> OpenCL s ()
runKernel source entryPoint args globalOffsets globalSizes localSizes =
  do queue <- getCommandQueue
     kernel <- getKernel source entryPoint
     let go [] real_args = do
           CL.setKernelArgs kernel (real_args [])
           ev <- CL.enqueueNDRangeKernel queue kernel globalOffsets globalSizes localSizes []
           CL.waitForEvents [ev]
         go (KernelArg f : fs) real_args = f (\arg -> go fs (real_args . (arg++)))
     liftIO $ go args id
