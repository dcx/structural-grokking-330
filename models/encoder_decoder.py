import torch
import torch.nn


def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int):
    lengths = lengths.long()
    # TODO: [KM] cannot figure this one out.
    input = torch.cat((input, torch.zeros_like(input[0:1])), dim=0)
    input.scatter_(0, lengths.unsqueeze(0).long(), value=eos_id)
    return input

def add_eos_pack(input: torch.Tensor, lengths: torch.Tensor, eos_id: int, sos_id: int):
    # effectively in_lens + 1
    lengths = (lengths - 1).long()

    # shift input to the right by one place
    input = torch.cat((input[1:, :], torch.zeros_like(input[0:1])), dim=0)

    # all SOS are now in place of EOS
    input[input == sos_id] = eos_id

    # add last EOS
    input.scatter_(0, lengths.unsqueeze(0).long(), value=eos_id)
    return input

