{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module AI.Funn.Flat.ParBox  where

import           Control.Applicative
import qualified Data.Binary as LB
import           Data.Proxy
import           Data.Type.Equality ((:~:)(..))
import           Foreign.C
import           Foreign.Ptr
import           GHC.TypeLits

import           AI.Funn.Common
import           AI.Funn.TypeLits
import           AI.Funn.Flat.Blob


data ParBox where
  ParBox :: KnownNat n => Blob n -> ParBox

instance LB.Binary ParBox where
  put (ParBox (b :: Blob n)) = do
    LB.put (natVal (Proxy @ n))
    LB.put b
  get = do
    n <- LB.get
    withNat n $ \(Proxy :: Proxy n) -> do
      (b :: Blob n) <- LB.get
      return (ParBox b)

openParBox :: forall n. KnownNat n => ParBox -> Maybe (Blob n)
openParBox (ParBox (b :: Blob m)) =
  case sameNat (Proxy @ n) (Proxy @ m) of
    Just Refl -> Just b
    Nothing -> Nothing
