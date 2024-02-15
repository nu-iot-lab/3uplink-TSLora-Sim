# EU spectrum 863 - 870 MHz, LoraWAN IoT
# 868000000, 868200000, 868400000 - join request, join accept
# 867000000, 867200000, 867400000, 867600000, 867800000, 869525000(10%) - SF
# https://www.thethingsnetwork.org/docs/lorawan/frequency-plans/
class Channels:
    # Channels
    Channel = {
        1: 867000000,  # SF 7
        2: 867200000,  # SF 8
        3: 867400000,  # SF 9
        4: 867600000,  # SF 10
        5: 867800000,  # SF 11
        6: 868400000,  # SF 12
        7: 868000000,  # join request/accept
    }

    @classmethod
    def get_sf_freq(cls, sf):
        return cls.Channel[sf - 6]

    @classmethod
    def get_jr_freq(cls):
        return cls.Channel[7]
