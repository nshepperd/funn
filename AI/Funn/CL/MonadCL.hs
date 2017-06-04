{-# LANGUAGE DeriveFunctor #-}
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
  OpenCL, runOpenCL, MonadCL(..), liftOpenCL,
  KernelArg(..), runKernel,
  doubleArg, int32Arg,
  Global, runOpenCLGlobal) where

import           Control.Applicative
import           Control.Monad
import           Control.Monad.Trans.Free
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

data ProgramCache = ProgramCache {
  _programs :: Map String CL.Program,
  _kernels :: Map (String, String) CL.Kernel
  }

data OpenCLF s r = GetPlatformID (CL.PlatformID -> r)
                 | GetContext (CL.Context -> r)
                 | GetDeviceID (CL.DeviceID -> r)
                 | GetCommandQueue (CL.CommandQueue -> r)
                 | GetKernel String String String (CL.Kernel -> r)
                 deriving (Functor)

newtype OpenCLT s m a = OpenCL (FreeT (OpenCLF s) m a)
  deriving (Functor, Applicative, Monad, MonadIO, MonadTrans)
type OpenCL s = OpenCLT s IO

class MonadIO m => MonadCL s m | m -> s where
  getPlatformID :: m CL.PlatformID
  getContext :: m CL.Context
  getDeviceID :: m CL.DeviceID
  getCommandQueue :: m CL.CommandQueue
  getKernel :: String -> String -> String -> m CL.Kernel

instance MonadIO m => MonadCL s (OpenCLT s m) where
  getPlatformID = OpenCL (liftF $ GetPlatformID id)
  getContext = OpenCL (liftF $ GetContext id)
  getDeviceID = OpenCL (liftF $ GetDeviceID id)
  getCommandQueue = OpenCL (liftF $ GetCommandQueue id)
  getKernel key src entry = OpenCL (liftF $ GetKernel key src entry id)

initialize :: IO CLInfo
initialize = do plat:_ <- CL.getPlatformIDs
                [dev] <- CL.getDeviceIDs [CL.DeviceTypeAll] plat
                ctx <- CL.createContext [dev] [CL.ContextPlatform plat] CL.NoContextCallback
                q <- CL.createCommandQueue ctx dev []
                return (CLInfo plat ctx dev q)

runOpenCL' :: CLInfo -> ProgramCache -> OpenCL s a -> IO (a, ProgramCache)
runOpenCL' (CLInfo platformID context device queue) cache (OpenCL k) = do
  go (_programs cache) (_kernels cache) k
    where
      go programs kernels (FreeT m) = do
        f <- m
        case f of
          Pure a -> return (a, ProgramCache programs kernels)
          Free (GetPlatformID k) -> go programs kernels (k platformID)
          Free (GetContext k) -> go programs kernels (k context)
          Free (GetDeviceID k) -> go programs kernels (k device)
          Free (GetCommandQueue k) -> go programs kernels (k queue)
          Free (GetKernel key src entry k) ->
            case Map.lookup (key, entry) kernels of
              Just kernel -> go programs kernels (k kernel)
              Nothing -> do (p, programs') <- getProgramLazy programs key src
                            kernel <- CL.createKernel p entry
                            let kernels' = Map.insert (key, entry) kernel kernels
                            go programs' kernels' (k kernel)
      getProgramLazy programs key src =
        case Map.lookup key programs of
          Just p -> return (p, programs)
          Nothing -> do p <- CL.createProgram context src
                        CL.buildProgram p [device] ""
                        return (p, Map.insert key p programs)

runOpenCL :: (forall s. OpenCL s a) -> IO a
runOpenCL cl = do
  clinfo <- initialize
  (a, _) <- runOpenCL' clinfo (ProgramCache Map.empty Map.empty) cl
  return a

liftOpenCL :: (MonadCL s m) => OpenCL s a -> m a
liftOpenCL (OpenCL (FreeT m)) = do
  free <- liftIO m
  case free of
    Pure a -> pure a
    Free (GetPlatformID k) -> getPlatformID >>= liftOpenCL . OpenCL . k
    Free (GetContext k) -> getContext >>= liftOpenCL . OpenCL . k
    Free (GetDeviceID k) -> getDeviceID >>= liftOpenCL . OpenCL . k
    Free (GetCommandQueue k) -> getCommandQueue >>= liftOpenCL . OpenCL . k
    Free (GetKernel key src entry k) -> getKernel key src entry >>= liftOpenCL . OpenCL . k

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
runOpenCLGlobal opencl = do
  context <- readIORef global_context
  info <- case context of
            Just info -> pure info
            Nothing -> do clinfo <- initialize
                          return (clinfo, (ProgramCache Map.empty Map.empty))
  go info
  where
    go (clinfo, programcache) = do
      (a, newpc) <- runOpenCL' clinfo programcache opencl
      writeIORef global_context (Just (clinfo, newpc))
      return a
