/// Defining variable for checking Network
var targetInterface: NIONetworkInterface? = nil

/** Check interface
 if interface is not available - error occurs
*/
if let interfaceAddress = CommandLine.arguments.dropFirst().first,
   let targetAddress = try? SocketAddress(ipAddress: interfaceAddress, port: 0) {
    for interface in try! System.enumerateInterfaces() {
        if interface.address == targetAddress {
            targetInterface = interface
            break
        }
    }

    if targetInterface == nil {
        fatalError("Could not find interface for \(interfaceAddress)")
    }
}
