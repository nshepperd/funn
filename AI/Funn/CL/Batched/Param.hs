{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.CL.Batched.Param (
  Param(..),
  reshape,
  split,
  appendD
  ) where


import           Control.Applicative
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe

import           AI.Funn.CL.Blob (Blob, BlobT(Blob))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as T
import qualified AI.Funn.CL.TensorLazy as TL
import qualified AI.Funn.CL.LazyMem as LM
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.Space

newtype Param (ω :: Nat) (n :: Nat) = Param { getParam :: Tensor '[n] }

instance Derivable (Param ω n) where
  type D (Param ω n) = TL.Tensor '[ω, n]

instance (MonadIO m, KnownNat n) => Zero m (Param ω n) where
  zero = Param <$> zero

instance (MonadIO m, KnownNat n) => Semi m (Param ω n) where
  plus (Param x) (Param y) = Param <$> plus x y

instance (MonadIO m, KnownNat n) => Additive m (Param ω n) where
  plusm xs = Param <$> plusm (map getParam xs)

instance (MonadIO m, KnownNat n) => Scale m Double (Param ω n) where
  scale x (Param xs) = Param <$> scale x xs

instance (MonadIO m, KnownNat n) => VectorSpace m Double (Param ω n) where
  {}

instance (MonadIO m, KnownNat n) => Inner m Double (Param ω n) where
  inner (Param x) (Param y) = inner x y

instance (MonadIO m, KnownNat n) => Finite m Double (Param ω n) where
  getBasis (Param x) = getBasis x

-- O(1)
reshape :: (Prod ds ~ n) => Param ω n -> Tensor ds
reshape (Param xs) = T.reshape xs

-- O(1)
split :: (KnownNat a, KnownNat b) => Param ω (a+b) -> (Param ω a, Param ω b)
split (Param xs) = case T.split xs of
                     (a, b) -> (Param a, Param b)

-- O(ω)
appendD :: forall ω a b. (KnownDimsF [ω, a, b]) => TL.Tensor [ω, a] -> TL.Tensor [ω, b] -> TL.Tensor [ω, a+b]
appendD = TL.appendW
