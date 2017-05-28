{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.CL.MonadCL (
  OpenCL, runOpenCL, MonadCL(..),
  KernelArg(..), runKernel,
  doubleArg, int32Arg,
  Global, runOpenCLGlobal) where

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
import           Data.IORef

import qualified Foreign.OpenCL.Bindings as CL
import qualified Foreign.OpenCL.Bindings.Synchronization as CL

data CLInfo = CLInfo {
  _platformID :: CL.PlatformID,
  _context :: CL.Context,
  _device :: CL.DeviceID,
  _queue :: CL.CommandQueue
  }
platformID :: Lens' CLInfo CL.PlatformID
platformID f (CLInfo a b c d) = (\a' -> CLInfo a' b c d) <$> f a
context :: Lens' CLInfo CL.Context
context f (CLInfo a b c d) = (\b' -> CLInfo a b' c d) <$> f b
device :: Lens' CLInfo CL.DeviceID
device f (CLInfo a b c d) = (\c' -> CLInfo a b c' d) <$> f c
queue :: Lens' CLInfo CL.CommandQueue
queue f (CLInfo a b c d) = (\d' -> CLInfo a b c d') <$> f d

data ProgramCache = ProgramCache {
  _programs :: Map String CL.Program,
  _kernels :: Map (String, String) CL.Kernel
  }
programs :: Lens' ProgramCache (Map String CL.Program)
programs f (ProgramCache x y) = (\x' -> ProgramCache x' y) <$> f x
kernels :: Lens' ProgramCache (Map (String,String) CL.Kernel)
kernels f (ProgramCache x y) = ProgramCache x <$> (f y)

class MonadIO m => MonadCL s m | m -> s where
  getPlatformID :: m CL.PlatformID
  getContext :: m CL.Context
  getDeviceID :: m CL.DeviceID
  getCommandQueue :: m CL.CommandQueue
  getKernel :: String -> String -> String -> m CL.Kernel

newtype OpenCLT s m a = OpenCL { getOpenCL :: ReaderT CLInfo (StateT ProgramCache m) a }
                      deriving (Functor, Applicative, Monad, MonadIO)

instance MonadTrans (OpenCLT s) where
  lift m = OpenCL (lift (lift m))

type OpenCL s = OpenCLT s IO

initialize :: IO CLInfo
initialize = do plat:_ <- CL.getPlatformIDs
                [dev] <- CL.getDeviceIDs [CL.DeviceTypeAll] plat
                ctx <- CL.createContext [dev] [CL.ContextPlatform plat] CL.NoContextCallback
                q <- CL.createCommandQueue ctx dev []
                return (CLInfo plat ctx dev q)

runOpenCL :: (forall s. OpenCL s a) -> IO a
runOpenCL (OpenCL k) = do clinfo <- initialize
                          evalStateT (runReaderT k clinfo) (ProgramCache Map.empty Map.empty)

instance MonadIO m => MonadCL s (OpenCLT s m) where
  getPlatformID = OpenCL (view platformID)
  getContext = OpenCL (view context)
  getDeviceID = OpenCL (view device)
  getCommandQueue = OpenCL (view queue)
  getKernel = getKernelCL

getProgram :: MonadIO m => String -> String -> OpenCLT s m CL.Program
getProgram key source = do ctx <- OpenCL $ view context
                           dev <- OpenCL $ view device
                           prg <- OpenCL $ use (programs . at key)
                           case prg of
                             Just program -> return program
                             Nothing -> do program <- liftIO $ CL.createProgram ctx source
                                           liftIO $ CL.buildProgram program [dev] ""
                                           OpenCL (programs . at key .= Just program)
                                           return program

createKernel :: MonadIO m => String -> String -> String -> OpenCLT s m CL.Kernel
createKernel key source entryPoint = do program <- getProgram key source
                                        liftIO $ CL.createKernel program entryPoint

getKernelCL :: MonadIO m => String -> String -> String -> OpenCLT s m CL.Kernel
getKernelCL key source entryPoint = do k <- OpenCL $ use (kernels . at (key,entryPoint))
                                       case k of
                                         Just kernel -> return kernel
                                         Nothing -> do kernel <- createKernel key source entryPoint
                                                       OpenCL (kernels . at (key,entryPoint) .= Just kernel)
                                                       return kernel

newtype KernelArg s = KernelArg (([CL.KernelArg] -> IO ()) -> IO ())

instance Monoid (KernelArg s) where
  mempty = KernelArg (\k -> k [])
  mappend (KernelArg f1) (KernelArg f2) = KernelArg $ \k ->
    f1 (\args1 ->
          f2 (\args2 -> k (args1 ++ args2)))


doubleArg :: Double -> KernelArg s
doubleArg x = KernelArg run
  where
    run f = f [CL.VArg x]

int32Arg :: Integral a => a -> KernelArg s
int32Arg a = KernelArg run
  where
    run f = f [CL.VArg (fromIntegral a :: Int32)]

runKernel :: MonadCL s m => String -> String  -> [KernelArg s] -> [CL.ClSize] -> [CL.ClSize] -> [CL.ClSize] -> m ()
runKernel source entryPoint args globalOffsets globalSizes localSizes =
  do queue <- getCommandQueue
     kernel <- getKernel source source entryPoint
     let go [] real_args = do
           CL.setKernelArgs kernel (real_args [])
           ev <- CL.enqueueNDRangeKernel queue kernel globalOffsets globalSizes localSizes []
           CL.waitForEvents [ev]
         go (KernelArg f : fs) real_args = f (\arg -> go fs (real_args . (arg++)))
     liftIO $ go args id

-- Run OpenCL in a global context.
-- More performant for tests.

data Global

global_context :: IORef (Maybe (CLInfo, ProgramCache))
global_context = unsafePerformIO $ newIORef Nothing

runOpenCLGlobal :: OpenCL Global a -> IO a
runOpenCLGlobal (OpenCL k) = do
  context <- readIORef global_context
  info <- case context of
            Just info -> pure info
            Nothing -> do clinfo <- initialize
                          return (clinfo, (ProgramCache Map.empty Map.empty))
  go info
  where
    go (clinfo, programcache) = do
      (a, newpc) <- runStateT (runReaderT k clinfo) programcache
      writeIORef global_context (Just (clinfo, newpc))
      return a
