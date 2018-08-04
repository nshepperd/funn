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
module AI.Funn.CL.Param (
  Param(..),
  reshape,
  split
  ) where


import           Control.Applicative
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import qualified Data.Foldable as F
import           Data.Foldable hiding (toList)
import           Data.IORef
import           Data.List hiding (replicate)
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           GHC.TypeLits
import           Prelude hiding (replicate)
import           System.IO.Unsafe

import           AI.Funn.CL.Blob (Blob, BlobT(Blob))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as T
import qualified AI.Funn.CL.TensorLazy as TL
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.Space

newtype Param (n :: Nat) = Param { getParam :: Tensor '[n] }

instance Derivable (Param n) where
  type D (Param n) = TL.Tensor '[n]

instance (MonadIO m, KnownNat n) => Zero m (Param n) where
  zero = Param <$> zero

instance (MonadIO m, KnownNat n) => Semi m (Param n) where
  plus (Param x) (Param y) = Param <$> plus x y

instance (MonadIO m, KnownNat n) => Additive m (Param n) where
  plusm xs = Param <$> plusm (map getParam . F.toList $ xs)

instance (MonadIO m, KnownNat n) => Scale m Double (Param n) where
  scale x (Param xs) = Param <$> scale x xs

instance (MonadIO m, KnownNat n) => VectorSpace m Double (Param n) where
  {}

instance (MonadIO m, KnownNat n) => Inner m Double (Param n) where
  inner (Param x) (Param y) = inner x y

instance (MonadIO m, KnownNat n) => Finite m Double (Param n) where
  getBasis (Param x) = getBasis x

-- O(1)
reshape :: (Prod ds ~ n) => Param n -> Tensor ds
reshape (Param xs) = T.reshape xs

-- O(1)
split :: (KnownNat a, KnownNat b) => Param (a+b) -> (Param a, Param b)
split (Param xs) = case T.split xs of
                     (a, b) -> (Param a, Param b)



-- split3 :: (KnownDims a, KnownDims b, KnownDims c)
--        => Tensor '[Prod a + Prod b + Prod c] -> (Tensor a, Tensor b, Tensor c)
-- split3 t123 = (reshape t1, reshape t2, reshape t3)
--   where
--     (t1, t23) = split t123
--     (t2, t3) = split t23

-- split4 :: (KnownDims a, KnownDims b, KnownDims c, KnownDims d)
--        => Tensor '[Prod a + Prod b + Prod c + Prod d]
--        -> (Tensor a, Tensor b, Tensor c, Tensor d)
-- split4 t1234 = (reshape t1, t2, t3, t4)
--   where
--     (t1, t234) = split t1234
--     (t2, t3, t4) = split3 t234

-- split5 :: (KnownDims a, KnownDims b, KnownDims c, KnownDims d, KnownDims e)
--        => Tensor '[Prod a + Prod b + Prod c + Prod d + Prod e]
--        -> (Tensor a, Tensor b, Tensor c, Tensor d, Tensor e)
-- split5 whole = (reshape t1, t2, t3, t4, t5)
--   where
--     (t1, rest) = split whole
--     (t2, t3, t4, t5) = split4 rest

-- append3 :: (KnownDims a, KnownDims b, KnownDims c)
--         => (Tensor a, Tensor b, Tensor c)
--         -> Tensor '[Prod a + Prod b + Prod c]
-- append3 (t1, t2, t3) = reshape t1 `append` reshape t2 `append` reshape t3

-- append4 :: (KnownDims a, KnownDims b, KnownDims c, KnownDims d)
--         => (Tensor a, Tensor b, Tensor c, Tensor d)
--         -> Tensor '[Prod a + Prod b + Prod c + Prod d]
-- append4 (t1, t2, t3, t4) = append (reshape t1) (append3 (t2, t3, t4))

-- append5 :: (KnownDims a, KnownDims b, KnownDims c, KnownDims d, KnownDims e)
--         => (Tensor a, Tensor b, Tensor c, Tensor d, Tensor e)
--         -> Tensor '[Prod a + Prod b + Prod c + Prod d + Prod e]
-- append5 (t1, t2, t3, t4, t5) = append (reshape t1) (append4 (t2, t3, t4, t5))
