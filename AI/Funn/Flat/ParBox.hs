{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module AI.Funn.Flat.ParBox  where

import qualified Codec.CBOR.Decoding as C
import qualified Codec.CBOR.Read as C
import qualified Codec.CBOR.Write as C
import qualified Codec.Serialise.Class as C
import           Control.Applicative
import qualified Data.ByteString.Lazy as LB
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

decodeOrError :: C.Serialise a => LB.ByteString -> a
decodeOrError bs = either (error . show) snd $ C.deserialiseFromBytes C.decode bs

encodeToByteString :: C.Serialise a => a -> LB.ByteString
encodeToByteString a = C.toLazyByteString $ C.encode a

instance C.Serialise ParBox where
  encode (ParBox (b :: Blob n)) =
    C.encode (natVal (Proxy @ n), b)
  decode = do
    2 <- C.decodeListLen
    n <- C.decode
    withNat n $ \(Proxy :: Proxy n) -> do
      (b :: Blob n) <- C.decode
      return (ParBox b)

openParBox :: forall n. KnownNat n => ParBox -> Maybe (Blob n)
openParBox (ParBox (b :: Blob m)) =
  case sameNat (Proxy @ n) (Proxy @ m) of
    Just Refl -> Just b
    Nothing -> Nothing
