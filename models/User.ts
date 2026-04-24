import { Schema, model, models } from "mongoose";

const UserSchema = new Schema(
  {
    username: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      index: true,
    },
    templateVersion: {
      type: Number,
      required: true,
      default: 2,
    },
    faceEmbedding: {
      type: [Number],
      required: true,
    },
    pulseSignature: {
      type: [Number],
      required: true,
    },
    rppgHrBpm: {
      type: Number,
      required: true,
    },
    bcgHrBpm: {
      type: Number,
      required: true,
      default: 0,
    },
    enrollmentCoherence: {
      type: Number,
      required: true,
      default: 0,
    },
  },
  {
    timestamps: true,
  }
);

const User = models.User || model("User", UserSchema);

export default User;
