import inet.node.ethernet.Eth10M;
import inet.node.ethernet.EthernetHost;
import inet.node.ethernet.EthernetSwitch;
import inet.physicallayer.wired.common.WireJunction;
import inet.tests.ethernet.EthernetHost2;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.StandardHost;
import inet.common.misc.ThruputMeteringChannel;
import inet.node.inet.Router;

network hw1_network
{
    parameters:
        int nodeCount = default(16);
        @display("bgb=6600,500");

    submodules:

        wireJunctions[nodeCount]: WireJunction {
            @display("p=100,200,r,50");
        }
        hosts[nodeCount]: EthernetHost {
            @display("p=100,300,r,50");
        }


    connections:
        for i=0..nodeCount-2 {
            wireJunctions[i].port++ <--> Eth10M <--> wireJunctions[i+1].port++;
            hosts[i].ethg <--> Eth10M <--> wireJunctions[i].port++;
        }
        hosts[nodeCount-1].ethg <--> Eth10M <--> wireJunctions[nodeCount-1].port++;
}

network hw1_network2
{
    parameters:
        int nodeCount = default(4);
        @display("bgb=6600,500");

    submodules:

        wireJunctions[nodeCount]: WireJunction {
            @display("p=100,200,r,50");
        }
        hosts[nodeCount]: EthernetHost {
            @display("p=100,300,r,50");
        }

        ethernetSwitch: EthernetSwitch {
            gates:
                ethg[2];
        }
    connections:
        for i=0..nodeCount-2, if i != int(nodeCount/2)-1 {
            wireJunctions[i].port++ <--> Eth10M <--> wireJunctions[i+1].port++;
            hosts[i].ethg <--> Eth10M <--> wireJunctions[i].port++;
        }
        for i=0..nodeCount-2, if i == int(nodeCount/2)-1 {
            wireJunctions[i].port++ <--> Eth10M <--> ethernetSwitch.ethg[0];
            wireJunctions[i+1].port++ <--> Eth10M <--> ethernetSwitch.ethg[1];
            hosts[i].ethg <--> Eth10M <--> wireJunctions[i].port++;
        }
        hosts[nodeCount-1].ethg <--> Eth10M <--> wireJunctions[nodeCount-1].port++;
}

network hw1_network_tcp
{
    types:
        channel tc1 extends ThruputMeteringChannel
        {
            thruputDisplayFormat = "B";
            datarate = 0.1Mbps;
        }
        channel tc2 extends ThruputMeteringChannel
        {
            thruputDisplayFormat = "B";
            //datarate = 1Mbps; // TCP - Q1
            datarate = 0.05Mbps; // TCP - Q2
        }
    submodules:
        configurator: Ipv4NetworkConfigurator {
        	@display("p=30,30");
        }
        client: StandardHost {
            @display("p=400,90");
        }
        router: Router {
            @display("p=300,90");
        }
        server: StandardHost {
            @display("p=200,90");
        }

    connections:
        client.pppg++ <--> tc1 <--> router.pppg++;
        server.pppg++ <--> tc2 <--> router.pppg++;
}

network hw1_network_tcp2
{
    types:
        channel tc1 extends ThruputMeteringChannel
        {
            thruputDisplayFormat = "B";
            datarate = 0.1Mbps;
        }
        channel tc2 extends ThruputMeteringChannel
        {
            thruputDisplayFormat = "B";
            //datarate = 0.1Mbps; // TCP - Q1
            datarate = 0.240Mbps; // TCP - Q2
        }
        channel tc3 extends ThruputMeteringChannel
        {
            thruputDisplayFormat = "B";
            datarate = 0.2Mbps;
        }
    submodules:
        configurator: Ipv4NetworkConfigurator {
        	@display("p=30,30");
        }
        client1: StandardHost {
            @display("p=400,60");
        }
        client2: StandardHost {
            @display("p=400,120");
        }
        router: Router {
            @display("p=300,90");
        }
        server: StandardHost {
            @display("p=200,90");
        }

    connections:
        client1.pppg++ <--> tc1 <--> router.pppg++;
        client2.pppg++ <--> tc3 <--> router.pppg++;
        server.pppg++ <--> tc2 <--> router.pppg++;
}



